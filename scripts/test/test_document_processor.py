from pathlib import Path

import pandas as pd
import pytest

from scripts.utilities.document_processor import DocumentProcessor


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def valid_rows() -> list[dict[str, object]]:
    return [
        {
            "document_id": "d1",
            "title": "First title",
            "text": "First document text.",
            "source": "FAO",
            "source_url": "https://example.test/1",
        },
        {
            "document_id": "d2",
            "title": "Second title",
            "text": "Second document text.",
            "source": "IITA",
            "source_url": "",
        },
    ]


def test_from_csv_loads_required_document_fields_and_lookup(tmp_path: Path) -> None:
    path = tmp_path / "documents.csv"
    write_csv(path, valid_rows())

    processor = DocumentProcessor.from_csv(path)
    document = processor.get("d1")

    assert document.document_id == "d1"
    assert document.title == "First title"
    assert document.text == "First document text."
    assert document.source == "FAO"
    assert document.source_url == "https://example.test/1"
    with pytest.raises(KeyError):
        processor.get("missing")


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"document_id": "d1", "title": "T", "source": "S", "source_url": ""}], "columns"),
        (valid_rows()[:1] + [valid_rows()[0]], "unique"),
        ([{**valid_rows()[0], "text": "  "}], "text"),
    ],
)
def test_from_csv_rejects_invalid_schema(rows: list[dict[str, object]], message: str, tmp_path: Path) -> None:
    path = tmp_path / "documents.csv"
    write_csv(path, rows)

    with pytest.raises(ValueError, match=message):
        DocumentProcessor.from_csv(path)


def test_context_is_bounded_deterministic_and_citation_complete(tmp_path: Path) -> None:
    path = tmp_path / "documents.csv"
    rows = valid_rows()
    rows[0]["text"] = "x" * 5000
    write_csv(path, rows)
    processor = DocumentProcessor.from_csv(path)
    first = processor.get("d1")
    second = processor.get("d2")

    context = processor.context([second, first])

    assert len(context) <= DocumentProcessor.MAX_CONTEXT_CHARS
    assert context == processor.context([second, first])
    assert context.index("Second title") < context.index("First title")
    assert "Source: IITA" in context
    assert "URL: https://example.test/1" in context
    assert "[2]" in context

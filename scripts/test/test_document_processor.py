from pathlib import Path

import pandas as pd
import pytest

from scripts.utilities.document_processor import DocumentProcessor
from scripts.utilities.models import RetrievedDocument


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


def test_public_metadata_and_count_are_safe_ui_views() -> None:
    processor = DocumentProcessor(
        [
            RetrievedDocument(
                "d1",
                "First title",
                "First document text.",
                0.0,
                1,
                "FAO",
                "https://example.test/1",
                "maize",
                "Kenya",
                "synthetic",
                "CC0",
            ),
            RetrievedDocument("d2", "Second title", "Second document text.", 0.0, 2),
        ]
    )

    metadata = processor.metadata

    assert processor.document_count == 2
    assert metadata["d1"] == {
        "title": "First title",
        "source": "FAO",
        "source_url": "https://example.test/1",
        "crop": "maize",
        "country": "Kenya",
        "origin": "synthetic",
        "license": "CC0",
    }
    assert metadata["d2"] == {"title": "Second title"}
    assert processor.document_metadata == metadata


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
    assert "d2" in context
    assert context.index("Second title") < context.index("First title")
    assert "d1" in context
    assert "Title: First title" in context
    assert "Source: FAO" in context
    assert "URL: https://example.test/1" in context
    assert "Source: IITA" in context
    assert "[2]" in context


def test_context_rejects_oversized_citation_headers_without_truncating_metadata() -> None:
    document = RetrievedDocument(
        document_id="doc-with-long-title",
        title="T" * 5990,
        text="body",
        score=0.0,
        rank=1,
        source="FAO",
        source_url="https://example.test/source",
    )

    with pytest.raises(ValueError, match="citation headers exceed context budget"):
        DocumentProcessor((document,)).context((document,))

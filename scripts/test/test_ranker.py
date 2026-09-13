from __future__ import annotations

import sys
import types

import pytest

from scripts.utilities.config import AppConfig
from scripts.utilities.document_processor import DocumentProcessor
from scripts.utilities.models import RetrievedDocument
from scripts.utilities.ranker import CrossEncoderRanker, Ranker, build_ranker


def processor() -> DocumentProcessor:
    return DocumentProcessor(
        (
            RetrievedDocument("d2", "Second title", "Second body", 0.0, 1),
            RetrievedDocument("d1", "First title", "First body", 0.0, 2),
            RetrievedDocument("d3", "Third title", "Third body", 0.0, 3),
        )
    )


def config(ranker: str = "cross_encoder") -> AppConfig:
    return AppConfig(
        api_base_url="https://example.test",
        api_key=None,
        model="answer-model",
        data_dir=None,  # type: ignore[arg-type]
        session_dir=None,  # type: ignore[arg-type]
        ranker=ranker,
        rank_model="configured-ranker-model",
        top_k=5,
    )


def install_fake_cross_encoder(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=lambda model_id: model),
    )


def test_cross_encoder_ranker_satisfies_ranker_protocol() -> None:
    assert isinstance(CrossEncoderRanker(processor(), "model"), Ranker)


def test_factory_uses_configured_cross_encoder_model_without_loading_weights() -> None:
    ranker = build_ranker(config(), processor())

    assert isinstance(ranker, CrossEncoderRanker)
    assert ranker.rank_model == "configured-ranker-model"
    assert ranker.model_loaded is False


def test_rank_lazily_loads_model_builds_pairs_and_returns_score_order() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] | None = None

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            self.pairs = pairs
            return [0.1, 0.9, 0.2]

    fake_model = FakeModel()
    ranker = CrossEncoderRanker(processor(), "model")
    assert ranker.model_loaded is False
    ranker._model = fake_model

    results = ranker.rank("how to farm", top_k=2)

    assert [document.document_id for document in results] == ["d1", "d3"]
    assert [document.score for document in results] == [0.9, 0.2]
    assert [document.rank for document in results] == [1, 2]
    assert fake_model.pairs == [
        ("how to farm", "Second title. Second body"),
        ("how to farm", "First title. First body"),
        ("how to farm", "Third title. Third body"),
    ]


def test_rank_loads_cross_encoder_only_on_first_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [1.0] * len(pairs)

    calls: list[str] = []

    def load(model_id: str) -> FakeModel:
        calls.append(model_id)
        return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=load),
    )
    ranker = CrossEncoderRanker(processor(), "local-cache-model")
    assert calls == []

    ranker.rank("query", top_k=1)
    ranker.rank("query", top_k=1)

    assert calls == ["local-cache-model"]


def test_rank_breaks_equal_scores_by_document_id_and_returns_unique_ids() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    ranker = CrossEncoderRanker(processor(), "model")
    ranker._model = FakeModel()

    results = ranker.rank("query", top_k=3)

    assert [document.document_id for document in results] == ["d1", "d2", "d3"]
    assert len({document.document_id for document in results}) == 3


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_rank_rejects_invalid_top_k(top_k: object) -> None:
    ranker = CrossEncoderRanker(processor(), "model")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        ranker.rank("query", top_k=top_k)  # type: ignore[arg-type]


def test_rank_caps_candidate_generation_without_changing_requested_top_k() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            assert len(pairs) == 2
            return [0.1, 0.9]

    ranker = CrossEncoderRanker(processor(), "model", candidate_k=2)
    ranker._model = FakeModel()

    results = ranker.rank("query", top_k=5)

    assert [document.document_id for document in results] == ["d1", "d2"]


@pytest.mark.parametrize("configured_ranker", ["lightgbm", "none"])
def test_factory_rejects_unimplemented_rankers(configured_ranker: str) -> None:
    with pytest.raises(ValueError, match="Unsupported ranker"):
        build_ranker(config(configured_ranker), processor())

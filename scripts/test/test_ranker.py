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


class FakeCandidateGenerator:
    def __init__(self, document_ids: list[str]) -> None:
        self.document_ids = document_ids
        self.calls: list[tuple[str, int]] = []

    def generate(self, query: str, candidate_k: int) -> list[str]:
        self.calls.append((query, candidate_k))
        return self.document_ids[:candidate_k]


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
        candidate_model="configured-candidate-model",
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
    assert ranker.candidate_model == "configured-candidate-model"
    assert ranker.model_loaded is False


def test_rank_lazily_loads_model_builds_pairs_and_returns_score_order() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] | None = None

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            self.pairs = pairs
            return [0.1, 0.9, 0.2]

    fake_model = FakeModel()
    ranker = CrossEncoderRanker(
        processor(), "model", candidate_generator=FakeCandidateGenerator(["d2", "d1", "d3"])
    )
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
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d2", "d1", "d3"]),
    )
    assert calls == []

    ranker.rank("query", top_k=1)
    ranker.rank("query", top_k=1)

    assert calls == ["local-cache-model"]


def test_rank_breaks_equal_scores_by_document_id_and_returns_unique_ids() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    ranker = CrossEncoderRanker(
        processor(), "model", candidate_generator=FakeCandidateGenerator(["d2", "d1", "d3"])
    )
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

    ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_k=2,
        candidate_generator=FakeCandidateGenerator(["d2", "d1", "d3"]),
    )
    ranker._model = FakeModel()

    results = ranker.rank("query", top_k=5)

    assert [document.document_id for document in results] == ["d1", "d2"]


@pytest.mark.parametrize("configured_ranker", ["lightgbm", "none"])
def test_factory_rejects_unimplemented_rankers(configured_ranker: str) -> None:
    with pytest.raises(ValueError, match="Unsupported ranker"):
        build_ranker(config(configured_ranker), processor())


def test_default_candidates_are_query_aware_and_not_corpus_order() -> None:
    documents = DocumentProcessor(
        (
            RetrievedDocument("d2", "Water storage", "Irrigation tanks", 0.0, 1),
            RetrievedDocument("d1", "Seed selection", "Choose resilient seed", 0.0, 2),
        )
    )
    ranker = CrossEncoderRanker(documents, "model", candidate_k=1)

    assert ranker._candidate_generator.generate("seed", 1) == ["d1"]


def test_injected_candidate_generator_receives_query_and_candidate_k() -> None:
    generator = FakeCandidateGenerator(["d3", "d1"])
    ranker = CrossEncoderRanker(processor(), "model", candidate_k=2, candidate_generator=generator)
    ranker._model = type("FakeModel", (), {"predict": lambda self, pairs: [0.1, 0.2]})()

    ranker.rank("farm query", top_k=1)

    assert generator.calls == [("farm query", 2)]


@pytest.mark.parametrize("query", ["", "   ", None, 1])
def test_rank_rejects_empty_or_non_string_query(query: object) -> None:
    ranker = CrossEncoderRanker(processor(), "model", candidate_generator=FakeCandidateGenerator([]))

    with pytest.raises(ValueError, match="query"):
        ranker.rank(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("candidate_k", [0, -1, True, 1.5])
def test_constructor_rejects_invalid_candidate_k(candidate_k: object) -> None:
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        CrossEncoderRanker(processor(), "model", candidate_k=candidate_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), "bad", True])
def test_rank_rejects_non_finite_or_invalid_scores(score: object) -> None:
    ranker = CrossEncoderRanker(
        processor(), "model", candidate_generator=FakeCandidateGenerator(["d1"])
    )
    ranker._model = type("FakeModel", (), {"predict": lambda self, pairs: [score]})()

    with pytest.raises(RuntimeError, match="score"):
        ranker.rank("query")


def test_rank_preserves_source_metadata() -> None:
    document = RetrievedDocument("d1", "Title", "Text", 0.0, 1, "manual", "https://source.test")
    ranker = CrossEncoderRanker(
        DocumentProcessor([document]),
        "model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
    )
    ranker._model = type("FakeModel", (), {"predict": lambda self, pairs: [0.5]})()

    result = ranker.rank("query")[0]

    assert (result.source, result.source_url) == ("manual", "https://source.test")


def test_rank_rejects_prediction_count_mismatch() -> None:
    ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_generator=FakeCandidateGenerator(["d1", "d2"]),
    )
    ranker._model = type("FakeModel", (), {"predict": lambda self, pairs: [0.5]})()

    with pytest.raises(RuntimeError, match="score count"):
        ranker.rank("query")


def test_rank_wraps_model_prediction_errors() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise ValueError("model failed")

    ranker = CrossEncoderRanker(
        processor(), "model", candidate_generator=FakeCandidateGenerator(["d1"])
    )
    ranker._model = FakeModel()

    with pytest.raises(RuntimeError, match="prediction failed"):
        ranker.rank("query")

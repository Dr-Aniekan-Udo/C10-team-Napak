from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.utilities.config import AppConfig
from scripts.utilities.document_processor import DocumentProcessor
from scripts.utilities.models import RetrievedDocument
from scripts.utilities.ranker import (
    DEFAULT_MODEL_LOAD_TIMEOUT,
    DEFAULT_PREDICTION_TIMEOUT,
    CrossEncoderRanker,
    Ranker,
    SparseCandidateGenerator,
    build_ranker,
)
from scripts.utilities.reranker_worker import RerankerWorkerError


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


class FakeWorker:
    def __init__(self, model: object) -> None:
        self.model = model
        self.start_calls: list[float] = []
        self.predict_calls: list[float] = []
        self.shutdown_calls = 0

    def start(self, timeout: float) -> None:
        self.start_calls.append(timeout)

    def predict(self, pairs: list[tuple[str, str]], timeout: float) -> object:
        self.predict_calls.append(timeout)
        return self.model.predict(pairs)  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


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


def test_cross_encoder_ranker_satisfies_ranker_protocol() -> None:
    assert isinstance(CrossEncoderRanker(processor(), "model"), Ranker)


def test_factory_uses_configured_cross_encoder_model_without_loading_weights() -> None:
    ranker = build_ranker(config(), processor())

    assert isinstance(ranker, CrossEncoderRanker)
    assert ranker.rank_model == "configured-ranker-model"
    assert ranker.candidate_model == "configured-candidate-model"
    assert isinstance(ranker._candidate_generator, SparseCandidateGenerator)
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

    def load(model_id: str) -> FakeWorker:
        calls.append(model_id)
        return FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d2", "d1", "d3"]),
        worker_factory=load,
    )
    assert calls == []

    ranker.rank("query", top_k=1)
    ranker.rank("query", top_k=1)

    assert calls == ["local-cache-model"]


def test_ranker_propagates_shared_load_and_prediction_timeouts() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    worker = FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=lambda model_name: worker,
    )

    ranker.rank("query", top_k=1)

    assert ranker.model_load_timeout == DEFAULT_MODEL_LOAD_TIMEOUT == 120.0
    assert ranker.prediction_timeout == DEFAULT_PREDICTION_TIMEOUT == 60.0
    assert worker.start_calls == [DEFAULT_MODEL_LOAD_TIMEOUT]
    assert worker.predict_calls == [DEFAULT_PREDICTION_TIMEOUT]


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


@pytest.mark.parametrize(
    ("candidate_ids", "message"),
    [
        (["d1", "d1"], "duplicate document IDs"),
        (["missing"], "unknown document ID"),
        ([1], "invalid document IDs"),
    ],
)
def test_rank_rejects_invalid_candidate_ids(
    candidate_ids: list[object], message: str
) -> None:
    ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_generator=FakeCandidateGenerator(candidate_ids),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match=message):
        ranker.rank("query")

    assert ranker.model_loaded is False


def test_empty_candidate_output_returns_without_loading_rank_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def load(model_id: str) -> FakeWorker:
        calls.append(model_id)
        return FakeWorker(object())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator([]),
        worker_factory=load,
    )

    assert ranker.rank("query") == []
    assert calls == []
    assert ranker.model_loaded is False


def test_thread_start_failure_transitions_to_unavailable_and_notifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
    )

    def fail_start(self: threading.Thread) -> None:
        raise RuntimeError("thread-start-secret")

    monkeypatch.setattr(threading.Thread, "start", fail_start)

    assert ranker.start_warmup() is False
    assert ranker.state == "unavailable"
    assert ranker.wait_until_ready(0.01) is False
    with pytest.raises(RuntimeError, match="warm-up could not start") as error:
        ranker.rank("query")
    assert "thread-start-secret" not in str(error.value)


def test_unexpected_warmup_base_exception_transitions_to_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FatalWarmup(BaseException):
        pass

    def load(model_id: str) -> object:
        raise FatalWarmup("warmup-secret")

    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=load,
    )

    assert ranker.start_warmup() is True
    assert ranker.wait_until_ready(1.0) is False
    assert ranker.state == "unavailable"
    with pytest.raises(RuntimeError, match="warm-up failed") as error:
        ranker.rank("query")
    assert "warmup-secret" not in str(error.value)


def test_direct_rank_starts_background_load_and_times_out_finitely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    def load(model_id: str) -> FakeWorker:
        started.set()
        assert release.wait(1.0)
        return FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        model_load_timeout=0.01,
        worker_factory=load,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ranker.rank, "query", 1)
        assert started.wait(1.0)
        with pytest.raises(RuntimeError, match="still loading"):
            future.result(timeout=0.5)
        assert ranker.state == "loading"
        release.set()
        assert ranker.wait_until_ready(1.0) is True


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_constructor_rejects_invalid_model_load_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="model_load_timeout"):
        CrossEncoderRanker(
            processor(),
            "model",
            model_load_timeout=timeout,  # type: ignore[arg-type]
        )


def test_wait_until_ready_rejects_huge_timeout_as_value_error() -> None:
    ranker = CrossEncoderRanker(processor(), "model")

    with pytest.raises(ValueError, match="timeout"):
        ranker.wait_until_ready(10**400)


def test_worker_prediction_timeout_terminalizes_and_shuts_down_worker() -> None:
    class FailingWorker:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def start(self, timeout: float) -> None:
            return None

        def predict(self, pairs: list[tuple[str, str]], timeout: float) -> object:
            raise RerankerWorkerError("secret", kind="prediction_timeout")

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    worker = FailingWorker()
    ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=lambda model_name: worker,
    )

    ranker.start_warmup()
    assert ranker.wait_until_ready(1.0) is True
    with pytest.raises(RuntimeError, match="prediction timed out"):
        ranker.rank("query")

    assert ranker.state == "unavailable"
    assert worker.shutdown_calls == 1


def test_public_ranker_shutdown_is_idempotent_and_prevents_late_reload() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    worker = FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=lambda model_name: worker,
    )

    assert ranker.start_warmup() is True
    assert ranker.wait_until_ready(1.0) is True
    ranker.shutdown()
    ranker.shutdown()

    assert ranker.model_loaded is False
    assert ranker.state == "unavailable"
    assert ranker.start_warmup() is False
    assert worker.shutdown_calls == 1


def test_shutdown_cancels_inflight_worker_and_joins_warmup_thread() -> None:
    started = threading.Event()
    release = threading.Event()

    class InFlightWorker:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def start(self, timeout: float) -> None:
            started.set()
            release.wait(1.0)

        def shutdown(self) -> None:
            self.shutdown_calls += 1
            release.set()

    worker = InFlightWorker()
    ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=lambda model_name: worker,
    )

    assert ranker.start_warmup() is True
    assert started.wait(1.0)
    ranker.shutdown(timeout=0.5)

    assert worker.shutdown_calls >= 1
    assert ranker._warmup_thread is None or not ranker._warmup_thread.is_alive()
    assert ranker.cleanup_error is None


@pytest.mark.parametrize("timeout", [threading.TIMEOUT_MAX + 1, 1e308, 10**400])
def test_ranker_rejects_timeouts_above_threading_max(timeout: object) -> None:
    with pytest.raises(ValueError, match="model_load_timeout"):
        CrossEncoderRanker(processor(), "model", model_load_timeout=timeout)  # type: ignore[arg-type]

    ranker = CrossEncoderRanker(processor(), "model")
    with pytest.raises(ValueError, match="timeout"):
        ranker.wait_until_ready(timeout)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout"):
        ranker.shutdown(timeout)  # type: ignore[arg-type]


def test_ranker_public_errors_have_no_raw_exception_cause() -> None:
    class FailingCandidates:
        def generate(self, query: str, candidate_k: int) -> list[str]:
            raise ValueError("candidate-secret")

    candidate_ranker = CrossEncoderRanker(
        processor(), "model", candidate_generator=FailingCandidates()
    )
    with pytest.raises(RuntimeError, match="Candidate generation failed") as candidate_error:
        candidate_ranker.rank("query")
    assert candidate_error.value.__cause__ is None

    class FailingModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise ValueError("prediction-secret")

    prediction_ranker = CrossEncoderRanker(
        processor(),
        "model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
    )
    prediction_ranker._model = FailingModel()
    with pytest.raises(RuntimeError, match="prediction failed") as prediction_error:
        prediction_ranker.rank("query")
    assert prediction_error.value.__cause__ is None


def test_warmup_lifecycle_is_guarded_and_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    def load(model_id: str) -> FakeWorker:
        calls.append(model_id)
        started.set()
        assert release.wait(1.0)
        return FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=load,
    )

    assert ranker.state == "not_started"
    assert ranker.start_warmup() is True
    assert started.wait(1.0)
    assert ranker.state == "loading"
    assert ranker.start_warmup() is False
    assert ranker.wait_until_ready(0.01) is False

    release.set()
    assert ranker.wait_until_ready(1.0) is True
    assert ranker.state == "ready"
    assert ranker.model_loaded is True
    assert ranker.start_warmup() is False
    assert len(calls) == 1


def test_concurrent_rank_calls_construct_one_cross_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.5] * len(pairs)

    def load(model_id: str) -> FakeWorker:
        calls.append(model_id)
        started.set()
        assert release.wait(1.0)
        return FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=load,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ranker.rank, "query", 1) for _ in range(4)]
        assert started.wait(1.0)
        assert ranker.state == "loading"
        release.set()
        results = [future.result(timeout=2.0) for future in futures]

    assert len(calls) == 1
    assert all([document.document_id for document in result] == ["d1"] for result in results)
    assert ranker.state == "ready"


def test_local_model_load_uses_cpu_torch_and_disables_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.75] * len(pairs)

    def load(model_id: str) -> FakeWorker:
        calls.append(model_id)
        return FakeWorker(FakeModel())
    ranker = CrossEncoderRanker(
        processor(),
        "local-cache-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=load,
    )

    results = ranker.rank("query")

    assert [document.document_id for document in results] == ["d1"]
    assert calls == ["local-cache-model"]


def test_local_cache_miss_becomes_unavailable_without_retry_or_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def load(model_id: str) -> object:
        calls.append(model_id)
        raise RuntimeError("private token and source text must not escape")
    ranker = CrossEncoderRanker(
        processor(),
        "missing-local-model",
        candidate_generator=FakeCandidateGenerator(["d1"]),
        worker_factory=load,
    )

    assert ranker.start_warmup() is True
    assert ranker.wait_until_ready(1.0) is False
    assert ranker.state == "unavailable"
    assert calls == ["missing-local-model"]

    with pytest.raises(RuntimeError, match="unavailable locally") as error:
        ranker.rank("query")
    assert "private token" not in str(error.value)
    assert calls == ["missing-local-model"]


def test_rank_fetches_each_candidate_document_once() -> None:
    class CountingProcessor(DocumentProcessor):
        def __init__(self) -> None:
            super().__init__(processor()._documents.values())
            self.lookups: list[str] = []

        def get(self, document_id: str) -> RetrievedDocument:
            self.lookups.append(document_id)
            return super().get(document_id)

    documents = CountingProcessor()
    ranker = CrossEncoderRanker(
        documents,
        "model",
        candidate_generator=FakeCandidateGenerator(["d2", "d1", "d3"]),
    )
    ranker._model = type(
        "FakeModel",
        (),
        {"predict": lambda self, pairs: [0.1, 0.9, 0.2]},
    )()

    results = ranker.rank("query", top_k=2)

    assert [document.document_id for document in results] == ["d1", "d3"]
    assert documents.lookups == ["d2", "d1", "d3"]

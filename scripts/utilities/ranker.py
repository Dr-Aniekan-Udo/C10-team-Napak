"""Cross-encoder ranking behind the application retrieval interface."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import AppConfig
from .document_processor import DocumentProcessor
from .models import RetrievedDocument
from .reranker_worker import (
    DEFAULT_CLEANUP_TIMEOUT,
    DEFAULT_MODEL_LOAD_TIMEOUT,
    DEFAULT_PREDICTION_TIMEOUT,
    RerankerWorkerClient,
    RerankerWorkerError,
)


@runtime_checkable
class Ranker(Protocol):
    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Return scored documents in deterministic rank order."""
        ...


class CandidateGenerator(Protocol):
    """Generate candidate IDs; dense or hybrid implementations are injectable."""

    def generate(self, query: str, candidate_k: int) -> Sequence[str]:
        """Return query-aware document IDs in candidate rank order."""
        ...


RerankerState = Literal["not_started", "loading", "ready", "unavailable"]


class SparseCandidateGenerator:
    """Query-aware sparse candidates matching Experiment 1 char-TF-IDF title-2."""

    def __init__(self, document_processor: DocumentProcessor) -> None:
        self.document_processor = document_processor
        self._document_ids = document_processor.document_ids
        self._vectorizer: TfidfVectorizer | None = None
        self._document_matrix: Any | None = None
        if self._document_ids:
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True
            )
            self._vectorizer = vectorizer
            texts = [
                " ".join(([document.title] * 2) + [document.text]).strip()
                for document in (document_processor.get(document_id) for document_id in self._document_ids)
            ]
            self._document_matrix = vectorizer.fit_transform(texts)

    def generate(self, query: str, candidate_k: int) -> list[str]:
        if not self._document_ids:
            return []
        assert self._vectorizer is not None
        assert self._document_matrix is not None
        scores = cosine_similarity(self._vectorizer.transform([query]), self._document_matrix)[0]
        order = np.lexsort((np.asarray(self._document_ids), -np.asarray(scores, dtype=float)))
        return [self._document_ids[index] for index in order[:candidate_k]]


class CrossEncoderRanker:
    """Score the processor corpus with a lazily loaded cross-encoder.

    Pair construction and ordering mirror Experiment 1's
    ``rerank_with_cross_encoder`` implementation. Model weights are loaded only
    when the first non-empty candidate set is ranked.

    The default candidate path is deterministic sparse char-TF-IDF so local and
    offline app construction never downloads a candidate model. ``candidate_model``
    records the configured dense/hybrid model for an injected
    :class:`CandidateGenerator`; it is not loaded or used by the default path.
    """

    def __init__(
        self,
        document_processor: DocumentProcessor,
        rank_model: str,
        candidate_k: int = 100,
        candidate_generator: CandidateGenerator | None = None,
        candidate_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_load_timeout: float = DEFAULT_MODEL_LOAD_TIMEOUT,
        prediction_timeout: float = DEFAULT_PREDICTION_TIMEOUT,
        worker_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(rank_model, str) or not rank_model.strip():
            raise ValueError("rank_model must be a non-empty string")
        self.document_processor = document_processor
        self.rank_model = rank_model.strip()
        self.candidate_k = _positive_integer(candidate_k, "candidate_k")
        self.model_load_timeout = _positive_finite_timeout(model_load_timeout, "model_load_timeout")
        self.prediction_timeout = _positive_finite_timeout(
            prediction_timeout, "prediction_timeout"
        )
        if worker_factory is not None and not callable(worker_factory):
            raise ValueError("worker_factory must be callable")
        self._worker_factory = worker_factory or self._default_worker_factory
        if not isinstance(candidate_model, str) or not candidate_model.strip():
            raise ValueError("candidate_model must be a non-empty string")
        # Keep candidate model provenance separate from the app-safe default
        # generator. Dense or hybrid generation must be injected explicitly.
        self.candidate_model = candidate_model.strip()
        self._candidate_generator = (
            candidate_generator
            if candidate_generator is not None
            else SparseCandidateGenerator(document_processor)
        )
        self._model: Any | None = None
        self._model_is_worker = False
        self._state_condition = threading.Condition()
        self._state: RerankerState = "not_started"
        self._unavailable_message: str | None = None
        self._warmup_thread: threading.Thread | None = None
        self._inflight_worker: object | None = None
        self._cleanup_error: str | None = None
        self._closed = False

        # Cache immutable corpus views used on every rank call. The default
        # candidate generator remains sparse char-TF-IDF with candidate_k=100.
        self._document_ids = tuple(document_processor.document_ids)
        self._document_id_set = frozenset(self._document_ids)

    @property
    def state(self) -> RerankerState:
        """Return current guarded model lifecycle state."""

        stale_worker: object | None = None
        with self._state_condition:
            if self._model is not None:
                if self._model_is_worker and not _worker_is_ready(self._model):
                    stale_worker = self._model
                    self._model = None
                    self._model_is_worker = False
                    self._set_unavailable_locked(
                        "Cross-encoder reranker worker exited unexpectedly; retry after restart."
                    )
                else:
                    return "ready"
            state = self._state
        if stale_worker is not None:
            _shutdown_worker_safely(stale_worker)
        return state

    @property
    def model_loaded(self) -> bool:
        return self.state == "ready"

    @property
    def cleanup_error(self) -> str | None:
        """Return sanitized teardown diagnostic, if ownership remains active."""

        return self._cleanup_error

    def start_warmup(self) -> bool:
        """Start one background local model load; return whether it started."""

        with self._state_condition:
            return self._start_warmup_locked()

    def wait_until_ready(self, timeout: float = 0.0) -> bool:
        """Wait at most ``timeout`` seconds for a ready model."""

        timeout = _nonnegative_finite_timeout(timeout, "timeout")
        if self.state == "ready":
            return True
        deadline = time.monotonic() + timeout
        with self._state_condition:
            while True:
                if self._model is not None or self._state == "ready":
                    return True
                if self._state in {"not_started", "unavailable"}:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state_condition.wait(remaining)

    def shutdown(self, timeout: float = DEFAULT_CLEANUP_TIMEOUT) -> None:
        """Stop owned worker and prevent late warm-up completion from reviving it."""

        timeout = _positive_finite_timeout(timeout, "timeout")
        deadline = time.monotonic() + timeout
        worker: object | None = None
        warmup_thread: threading.Thread | None = None
        with self._state_condition:
            self._closed = True
            if self._model_is_worker:
                worker = self._model
            elif self._inflight_worker is not None:
                worker = self._inflight_worker
            warmup_thread = self._warmup_thread
            self._model = None
            self._model_is_worker = False
            self._state = "unavailable"
            self._unavailable_message = "Cross-encoder reranker is shut down."
            self._state_condition.notify_all()
        worker_ok = _shutdown_worker_safely(
            worker,
            timeout=max(0.0, deadline - time.monotonic()),
        )
        thread_ok = True
        if warmup_thread is not None:
            thread_ok = _join_thread_safely(
                warmup_thread,
                max(0.0, deadline - time.monotonic()),
            )
        with self._state_condition:
            active_thread = self._warmup_thread
            active_worker = self._inflight_worker
            if active_thread is not None and not active_thread.is_alive():
                self._warmup_thread = None
                active_thread = None
            if worker_ok and thread_ok and active_thread is None and active_worker is None:
                self._cleanup_error = None
            else:
                self._cleanup_error = "Cross-encoder reranker cleanup failed."

    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        top_k = _positive_integer(top_k, "top_k")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            candidate_ids = list(self._candidate_generator.generate(query, self.candidate_k))
        except Exception:
            raise RuntimeError("Candidate generation failed.") from None
        if not candidate_ids:
            return []
        if not all(isinstance(document_id, str) for document_id in candidate_ids):
            raise RuntimeError("Candidate generator returned invalid document IDs.")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise RuntimeError("Candidate generator returned duplicate document IDs.")
        unknown = [document_id for document_id in candidate_ids if document_id not in self._document_id_set]
        if unknown:
            raise RuntimeError(f"Candidate generator returned unknown document ID: {unknown[0]}")
        documents = {
            document_id: self.document_processor.get(document_id)
            for document_id in candidate_ids
        }
        pairs = [
            (
                query,
                f"{documents[document_id].title}. {documents[document_id].text}",
            )
            for document_id in candidate_ids
        ]
        model = self._get_model()
        try:
            if self._model_is_worker:
                scores = model.predict(pairs, timeout=self.prediction_timeout)
            else:
                scores = model.predict(pairs)
        except RerankerWorkerError as exc:
            self._handle_worker_error(exc)
            raise RuntimeError(self._worker_failure_message(exc)) from None
        except Exception:
            raise RuntimeError("Cross-encoder prediction failed.") from None
        try:
            scores = list(scores)
        except Exception:
            raise RuntimeError("Cross-encoder returned invalid scores.") from None
        if len(scores) != len(candidate_ids):
            raise RuntimeError(
                "Cross-encoder returned a score count different from candidates."
            )

        scored = []
        for document_id, score in zip(candidate_ids, scores):
            if isinstance(score, bool) or not isinstance(score, (int, float, np.number)):
                raise RuntimeError("Cross-encoder returned an invalid score.")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RuntimeError("Cross-encoder returned a non-finite score.")
            scored.append((document_id, numeric_score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        results: list[RetrievedDocument] = []
        for rank, (document_id, score) in enumerate(scored[:top_k], 1):
            document = documents[document_id]
            results.append(
                RetrievedDocument(
                    document_id=document.document_id,
                    title=document.title,
                    text=document.text,
                    score=score,
                    rank=rank,
                    source=document.source,
                    source_url=document.source_url,
                    crop=document.crop,
                    country=document.country,
                    origin=document.origin,
                    license=document.license,
                )
            )
        return results

    def _get_model(self) -> Any:
        with self._state_condition:
            if self._closed:
                raise RuntimeError(self._unavailable_error()) from None
            if self._model is not None:
                self._state = "ready"
                return self._model
            if self._state == "unavailable":
                raise RuntimeError(self._unavailable_error()) from None
            if self._state == "not_started" and not self._start_warmup_locked():
                raise RuntimeError(self._unavailable_error())
            return self._wait_for_model_locked(self.model_load_timeout)

    def _warmup_worker(self) -> None:
        try:
            self._load_claimed_model(raise_on_error=False)
        except BaseException:
            self._finish_unavailable(
                "Cross-encoder reranker warm-up failed; retry after restart.",
                raise_on_error=False,
            )
        finally:
            with self._state_condition:
                if self._warmup_thread is threading.current_thread():
                    self._warmup_thread = None

    def _start_warmup_locked(self) -> bool:
        if self._closed:
            return False
        if self._model is not None:
            self._state = "ready"
            return False
        if self._state != "not_started":
            return False
        self._state = "loading"
        try:
            self._warmup_thread = threading.Thread(
                target=self._warmup_worker,
                name="cross-encoder-warmup",
                daemon=True,
            )
            self._warmup_thread.start()
        except BaseException:
            self._warmup_thread = None
            self._set_unavailable_locked(
                "Cross-encoder reranker warm-up could not start; retry after restart."
            )
            return False
        return True

    def _wait_for_model_locked(self, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            if self._model is not None:
                return self._model
            if self._state == "unavailable":
                raise RuntimeError(self._unavailable_error())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Cross-encoder reranker is still loading; retry after warm-up."
                ) from None
            self._state_condition.wait(remaining)

    def _load_claimed_model(self, *, raise_on_error: bool) -> Any | None:
        model: Any | None = None
        try:
            with self._state_condition:
                if self._closed:
                    return None
            model = self._worker_factory(self.rank_model)
            if model is None:
                raise RuntimeError("worker factory returned no worker")
            with self._state_condition:
                if self._closed:
                    should_shutdown = True
                else:
                    should_shutdown = False
                    self._inflight_worker = model
            if should_shutdown:
                _shutdown_worker_safely(model)
                with self._state_condition:
                    if self._inflight_worker is model:
                        self._inflight_worker = None
                return None
            model.start(self.model_load_timeout)
        except RerankerWorkerError as exc:
            _shutdown_worker_safely(model)
            with self._state_condition:
                if self._inflight_worker is model:
                    self._inflight_worker = None
            return self._finish_unavailable(
                self._worker_failure_message(exc),
                raise_on_error=raise_on_error,
            )
        except Exception:
            _shutdown_worker_safely(model)
            with self._state_condition:
                if self._inflight_worker is model:
                    self._inflight_worker = None
            return self._finish_unavailable(
                f"Cross-encoder model '{self.rank_model}' is unavailable locally; "
                "worker startup failed.",
                raise_on_error=raise_on_error,
            )
        except BaseException:
            _shutdown_worker_safely(model)
            with self._state_condition:
                if self._inflight_worker is model:
                    self._inflight_worker = None
            return self._finish_unavailable(
                "Cross-encoder reranker warm-up failed; retry after restart.",
                raise_on_error=raise_on_error,
            )

        with self._state_condition:
            if self._closed:
                should_shutdown = True
            else:
                should_shutdown = False
                if self._inflight_worker is model:
                    self._inflight_worker = None
                self._model = model
                self._model_is_worker = True
                self._state = "ready"
                self._state_condition.notify_all()
        if should_shutdown:
            _shutdown_worker_safely(model)
            with self._state_condition:
                if self._inflight_worker is model:
                    self._inflight_worker = None
            return None
        return model

    def _default_worker_factory(self, model_name: str) -> RerankerWorkerClient:
        return RerankerWorkerClient(
            model_name,
            prediction_timeout=self.prediction_timeout,
        )

    def _handle_worker_error(self, error: RerankerWorkerError) -> None:
        worker: object | None = None
        with self._state_condition:
            if self._model_is_worker:
                worker = self._model
            self._model = None
            self._model_is_worker = False
            self._set_unavailable_locked(self._worker_failure_message(error))
        _shutdown_worker_safely(worker)

    def _worker_failure_message(self, error: RerankerWorkerError) -> str:
        if error.kind == "prediction_timeout":
            return "Cross-encoder reranker prediction timed out; no result generated."
        if error.kind == "readiness_timeout":
            return "Cross-encoder reranker worker readiness timed out; no result generated."
        if error.kind == "start_timeout":
            return "Cross-encoder reranker worker start timed out; no result generated."
        if error.kind == "prediction":
            return "Cross-encoder reranker prediction failed; no result generated."
        if error.kind == "busy":
            return "Cross-encoder reranker is busy; retry the request."
        if error.kind == "cleanup":
            return "Cross-encoder reranker cleanup failed; no result generated."
        return (
            f"Cross-encoder model '{self.rank_model}' is unavailable locally; "
            "worker startup failed."
        )

    def _finish_unavailable(self, message: str, *, raise_on_error: bool) -> None:
        with self._state_condition:
            self._set_unavailable_locked(message)
        if raise_on_error:
            raise RuntimeError(message) from None
        return None

    def _set_unavailable_locked(self, message: str) -> None:
        if self._closed:
            return
        self._unavailable_message = message
        self._state = "unavailable"
        self._state_condition.notify_all()

    def _unavailable_error(self) -> str:
        return self._unavailable_message or (
            f"Cross-encoder model '{self.rank_model}' is unavailable locally."
        )


def build_ranker(config: AppConfig, processor: DocumentProcessor) -> Ranker:
    """Build ranker without weights; default candidates stay sparse and offline-safe.

    ``config.candidate_model`` is provenance/configuration for a separately
    injected dense or hybrid generator, not a request to load model weights here.
    """

    if config.ranker == "cross_encoder":
        return CrossEncoderRanker(
            processor,
            config.rank_model,
            candidate_model=config.candidate_model,
        )
    raise ValueError(f"Unsupported ranker: {config.ranker}")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite_timeout(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite positive number") from None
    if not math.isfinite(numeric) or numeric > threading.TIMEOUT_MAX:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def _nonnegative_finite_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite non-negative number") from None
    if not math.isfinite(numeric) or numeric < 0 or numeric > threading.TIMEOUT_MAX:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def _worker_is_ready(worker: object) -> bool:
    try:
        ready = getattr(worker, "ready")
    except AttributeError:
        # Injected test doubles and legacy ranker adapters have no health API.
        return True
    except BaseException:
        return False
    return bool(ready)


def _join_thread_safely(thread: threading.Thread, timeout: float) -> bool:
    if thread is threading.current_thread():
        return False
    try:
        thread.join(timeout=min(timeout, threading.TIMEOUT_MAX))
    except BaseException:
        return False
    try:
        return not thread.is_alive()
    except BaseException:
        return False


def _shutdown_worker_safely(
    worker: object | None,
    *,
    timeout: float | None = None,
) -> bool:
    if worker is None:
        return True
    try:
        shutdown = getattr(worker, "shutdown", None)
        if not callable(shutdown):
            return True
        if timeout is None:
            shutdown()
        else:
            try:
                shutdown(timeout=timeout)
            except TypeError:
                try:
                    shutdown(timeout)
                except TypeError:
                    # Preserve compatibility with injected ranker doubles that
                    # expose the original no-argument lifecycle method.
                    shutdown()
    except BaseException:
        return False
    try:
        return getattr(worker, "cleanup_error", None) is None
    except BaseException:
        return False

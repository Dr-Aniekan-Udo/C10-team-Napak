"""Cross-encoder ranking behind the application retrieval interface."""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import AppConfig
from .document_processor import DocumentProcessor
from .models import RetrievedDocument


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


class SparseCandidateGenerator:
    """Query-aware sparse candidates matching Experiment 1 char-TF-IDF title-2."""

    def __init__(self, document_processor: DocumentProcessor) -> None:
        self.document_processor = document_processor
        self._document_ids = sorted(document_processor._documents)
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
    ) -> None:
        if not isinstance(rank_model, str) or not rank_model.strip():
            raise ValueError("rank_model must be a non-empty string")
        self.document_processor = document_processor
        self.rank_model = rank_model.strip()
        self.candidate_k = _positive_integer(candidate_k, "candidate_k")
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

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        top_k = _positive_integer(top_k, "top_k")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            candidate_ids = list(self._candidate_generator.generate(query, self.candidate_k))
        except Exception as exc:
            raise RuntimeError("Candidate generation failed.") from exc
        if not candidate_ids:
            return []
        if not all(isinstance(document_id, str) for document_id in candidate_ids):
            raise RuntimeError("Candidate generator returned invalid document IDs.")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise RuntimeError("Candidate generator returned duplicate document IDs.")
        unknown = [
            document_id
            for document_id in candidate_ids
            if document_id not in self.document_processor._documents
        ]
        if unknown:
            raise RuntimeError(f"Candidate generator returned unknown document ID: {unknown[0]}")
        pairs = [
            (
                query,
                f"{self.document_processor.get(document_id).title}. "
                f"{self.document_processor.get(document_id).text}",
            )
            for document_id in candidate_ids
        ]
        try:
            scores = self._get_model().predict(pairs)
        except Exception as exc:
            raise RuntimeError("Cross-encoder prediction failed.") from exc
        try:
            scores = list(scores)
        except Exception as exc:
            raise RuntimeError("Cross-encoder returned invalid scores.") from exc
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
            document = self.document_processor.get(document_id)
            results.append(
                RetrievedDocument(
                    document_id=document.document_id,
                    title=document.title,
                    text=document.text,
                    score=score,
                    rank=rank,
                    source=document.source,
                    source_url=document.source_url,
                )
            )
        return results

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:
                raise RuntimeError(
                    "Cross-encoder requires sentence-transformers and Torch; "
                    "install the cross_encoder extra."
                ) from exc
            try:
                self._model = CrossEncoder(self.rank_model)
            except Exception as exc:
                raise RuntimeError(
                    f"Cross-encoder model '{self.rank_model}' could not be loaded."
                ) from exc
        return self._model


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

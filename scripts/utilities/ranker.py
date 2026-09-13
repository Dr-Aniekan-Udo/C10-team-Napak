"""Cross-encoder ranking behind the application retrieval interface."""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from .config import AppConfig
from .document_processor import DocumentProcessor
from .models import RetrievedDocument


@runtime_checkable
class Ranker(Protocol):
    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Return scored documents in deterministic rank order."""
        ...


class CrossEncoderRanker:
    """Score the processor corpus with a lazily loaded cross-encoder.

    Pair construction and ordering mirror Experiment 1's
    ``rerank_with_cross_encoder`` implementation. Model weights are loaded only
    when the first non-empty candidate set is ranked.
    """

    def __init__(
        self,
        document_processor: DocumentProcessor,
        rank_model: str,
        candidate_k: int = 100,
    ) -> None:
        if not isinstance(rank_model, str) or not rank_model.strip():
            raise ValueError("rank_model must be a non-empty string")
        self.document_processor = document_processor
        self.rank_model = rank_model.strip()
        self.candidate_k = _positive_integer(candidate_k, "candidate_k")
        self._model: Any | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        top_k = _positive_integer(top_k, "top_k")
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        candidate_ids = list(self.document_processor._documents)[: self.candidate_k]
        if not candidate_ids:
            return []
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
        scores = list(scores)
        if len(scores) != len(candidate_ids):
            raise RuntimeError(
                "Cross-encoder returned a score count different from candidates."
            )

        scored = [
            (document_id, float(score))
            for document_id, score in zip(candidate_ids, scores)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        results: list[RetrievedDocument] = []
        for rank, (document_id, score) in enumerate(scored[:top_k], 1):
            if not math.isfinite(score):
                raise RuntimeError("Cross-encoder returned a non-finite score.")
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
    """Build configured ranker without loading model weights."""

    if config.ranker == "cross_encoder":
        return CrossEncoderRanker(processor, config.rank_model)
    raise ValueError(f"Unsupported ranker: {config.ranker}")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value

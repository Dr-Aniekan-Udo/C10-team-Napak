"""Small deterministic helpers shared by retrieval experiment notebooks.

Functions in this module accept all experiment inputs explicitly. Importing it
does not load data, execute an experiment, or create an output directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILENAMES: Mapping[str, str] = {
    "documents": "documents.csv",
    "train_queries": "train_queries.csv",
    "qrels": "qrels_train.csv",
    "test_queries": "test_queries.csv",
    "sample_submission": "sample_submission.csv",
}

_DOCUMENT_COLUMNS = {"document_id", "title", "text"}
_TRAIN_QUERY_COLUMNS = {"query_id", "query"}
_QREL_COLUMNS = {"query_id", "document_id", "relevance"}
_TEST_QUERY_COLUMNS = {"query_id", "query"}
_SUBMISSION_COLUMNS = {"QueryId", "DocumentId"}


def _require_columns(
    frame: pd.DataFrame, frame_name: str, required: set[str]
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{frame_name} missing required columns: {sorted(missing)}"
        )


def _stringify_column(frame: pd.DataFrame, column: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{column!r} contains missing IDs")
    frame[column] = frame[column].astype(str)


def load_competition_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load supplied competition CSVs using the benchmark's column names.

    IDs are normalized to strings so rankings and qrels cannot diverge merely
    because pandas inferred one CSV column as numeric. No validation beyond
    required columns and ID normalization is performed here.
    """

    data_dir = Path(data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Competition data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Competition data path is not a directory: {data_dir}")

    tables = {
        name: pd.read_csv(data_dir / filename)
        for name, filename in DATA_FILENAMES.items()
    }
    required_columns = {
        "documents": _DOCUMENT_COLUMNS,
        "train_queries": _TRAIN_QUERY_COLUMNS,
        "qrels": _QREL_COLUMNS,
        "test_queries": _TEST_QUERY_COLUMNS,
        "sample_submission": _SUBMISSION_COLUMNS,
    }
    for name, columns in required_columns.items():
        _require_columns(tables[name], name, columns)

    for name in ("documents", "train_queries", "qrels", "test_queries"):
        if "query_id" in tables[name]:
            _stringify_column(tables[name], "query_id")
        if "document_id" in tables[name]:
            _stringify_column(tables[name], "document_id")
    _stringify_column(tables["sample_submission"], "QueryId")
    _stringify_column(tables["sample_submission"], "DocumentId")
    return tables


def _normalized_qrels(qrels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(qrels, "qrels", _QREL_COLUMNS)
    normalized = qrels[["query_id", "document_id", "relevance"]].copy()
    _stringify_column(normalized, "query_id")
    _stringify_column(normalized, "document_id")
    normalized["relevance"] = pd.to_numeric(
        normalized["relevance"], errors="raise"
    ).astype(float)
    if not normalized["relevance"].map(math.isfinite).all():
        raise ValueError("qrels relevance values must be finite")
    if (normalized["relevance"] < 0).any():
        raise ValueError("qrels relevance values must be non-negative")
    return normalized


def _normalized_train_queries(train_queries: pd.DataFrame) -> pd.DataFrame:
    _require_columns(train_queries, "train_queries", _TRAIN_QUERY_COLUMNS)
    normalized = train_queries[["query_id", "query"]].copy()
    _stringify_column(normalized, "query_id")
    if not normalized["query_id"].is_unique:
        raise ValueError("train_queries query_id values must be unique")
    return normalized


def _query_groups_from_positive_documents(
    train_queries: pd.DataFrame,
    qrels: pd.DataFrame,
    component_by_document: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    queries = _normalized_train_queries(train_queries)
    normalized_qrels = _normalized_qrels(qrels)
    query_ids = set(queries["query_id"])
    qrel_query_ids = set(normalized_qrels["query_id"])
    unknown_queries = qrel_query_ids - query_ids
    if unknown_queries:
        raise ValueError(
            "qrels contains query IDs absent from train_queries: "
            f"{sorted(unknown_queries)[:5]}"
        )

    positive = normalized_qrels.loc[normalized_qrels["relevance"] > 0]
    grouped_positive = {
        str(query_id): set(group["document_id"])
        for query_id, group in positive.groupby("query_id", sort=False)
    }

    query_components: dict[str, set[str]] = {}
    for query_id in queries["query_id"]:
        document_ids = grouped_positive.get(query_id, set())
        if component_by_document is not None:
            unknown_documents = document_ids - set(component_by_document)
            if unknown_documents:
                raise ValueError(
                    "qrels contains positive document IDs absent from documents: "
                    f"{sorted(unknown_documents)[:5]}"
                )
            document_ids = {
                component_by_document[document_id] for document_id in document_ids
            }
        query_components[query_id] = document_ids

    if component_by_document is None:
        return [
            (
                query_id,
                "|".join(sorted(query_components[query_id]))
                if query_components[query_id]
                else f"singleton:{query_id}",
            )
            for query_id in queries["query_id"]
        ]

    # Connected query groups are required here: two queries sharing one
    # positive component must remain on the same side even when their full
    # component sets differ.
    query_id_list = queries["query_id"].tolist()
    union_find = _UnionFind(len(query_id_list))
    first_query_by_component: dict[str, int] = {}
    for index, query_id in enumerate(query_id_list):
        for component in sorted(query_components[query_id]):
            first_index = first_query_by_component.setdefault(component, index)
            union_find.union(index, first_index)

    components_by_root: dict[int, set[str]] = {}
    for index, query_id in enumerate(query_id_list):
        root = union_find.find(index)
        components_by_root.setdefault(root, set()).update(
            query_components[query_id]
        )
    group_key_by_root = {
        root: "component_group:" + "|".join(sorted(components))
        for root, components in components_by_root.items()
    }
    groups: list[tuple[str, str]] = []
    for index, query_id in enumerate(query_id_list):
        if not query_components[query_id]:
            group_key = f"singleton:{query_id}"
        else:
            group_key = group_key_by_root[union_find.find(index)]
        groups.append((query_id, group_key))
    return groups


def _split_query_groups(
    query_groups: Sequence[tuple[str, str]], seed: int
) -> tuple[set[str], set[str]]:
    if not query_groups:
        raise ValueError("Cannot split empty train_queries")
    group_ids = sorted({group_key for _, group_key in query_groups})
    if len(group_ids) < 2:
        raise ValueError(
            "Cannot construct a valid split: fewer than two query groups"
        )

    ordered_groups = sorted(
        group_ids,
        key=lambda group: hashlib.md5(
            f"{seed}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = min(
        len(ordered_groups) - 1,
        max(1, round(len(ordered_groups) * 0.2)),
    )
    validation_groups = set(ordered_groups[-validation_count:])
    validation_ids = {
        query_id
        for query_id, group_key in query_groups
        if group_key in validation_groups
    }
    all_query_ids = {query_id for query_id, _ in query_groups}
    training_ids = all_query_ids - validation_ids
    if not training_ids or not validation_ids:
        raise ValueError("Cannot construct non-empty training and validation splits")
    if training_ids & validation_ids:
        raise AssertionError("query split sets overlap")
    return training_ids, validation_ids


def build_grouped_split(
    train_queries: pd.DataFrame,
    qrels: pd.DataFrame,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Build deterministic split keeping identical positive-document sets together."""

    return _split_query_groups(
        _query_groups_from_positive_documents(train_queries, qrels), seed
    )


def _normalize_document_text(title: object, text: object) -> str:
    title_text = "" if pd.isna(title) else str(title)
    body_text = "" if pd.isna(text) else str(text)
    combined = unicodedata.normalize("NFKC", f"{title_text} {body_text}").casefold()
    alphanumeric = "".join(
        character if character.isalnum() else " " for character in combined
    )
    return " ".join(alphanumeric.split())


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _document_components(
    documents: pd.DataFrame, cosine_threshold: float
) -> dict[str, str]:
    _require_columns(documents, "documents", _DOCUMENT_COLUMNS)
    if not math.isfinite(float(cosine_threshold)) or not 0 <= cosine_threshold <= 1:
        raise ValueError("cosine_threshold must be a finite value in [0, 1]")

    document_frame = documents[["document_id", "title", "text"]].copy()
    _stringify_column(document_frame, "document_id")
    if not document_frame["document_id"].is_unique:
        raise ValueError("documents document_id values must be unique")
    if document_frame.empty:
        raise ValueError("Cannot build document components from empty documents")

    document_ids = document_frame["document_id"].tolist()
    normalized_texts = [
        _normalize_document_text(title, text)
        for title, text in zip(document_frame["title"], document_frame["text"])
    ]
    union_find = _UnionFind(len(document_ids))

    first_by_text: dict[str, int] = {}
    for index, normalized_text in enumerate(normalized_texts):
        first_index = first_by_text.setdefault(normalized_text, index)
        union_find.union(index, first_index)

    nonempty_indices = [
        index for index, normalized_text in enumerate(normalized_texts) if normalized_text
    ]
    if nonempty_indices:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        matrix = vectorizer.fit_transform(normalized_texts)
        similarities = cosine_similarity(matrix)
        for left_index in range(len(document_ids)):
            for right_index in range(left_index + 1, len(document_ids)):
                if similarities[left_index, right_index] >= cosine_threshold:
                    union_find.union(left_index, right_index)

    members_by_root: dict[int, list[str]] = {}
    for index, document_id in enumerate(document_ids):
        members_by_root.setdefault(union_find.find(index), []).append(document_id)
    component_key_by_root = {
        root: "component:" + "|".join(sorted(member_ids))
        for root, member_ids in members_by_root.items()
    }
    return {
        document_id: component_key_by_root[union_find.find(index)]
        for index, document_id in enumerate(document_ids)
    }


def build_document_disjoint_split(
    train_queries: pd.DataFrame,
    qrels: pd.DataFrame,
    documents: pd.DataFrame,
    cosine_threshold: float = 0.90,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split query groups whose positive document components do not overlap.

    Components join exact normalized title/body duplicates and connected
    near-duplicate pairs at the supplied TF-IDF cosine threshold. A query's
    group is defined by its positive components, so holding out whole groups
    prevents positive components from crossing the split.
    """

    component_by_document = _document_components(documents, cosine_threshold)
    query_groups = _query_groups_from_positive_documents(
        train_queries, qrels, component_by_document
    )
    return _split_query_groups(query_groups, seed)


def _validate_ranked_document_ids(
    ranked_ids: Sequence[object], query_id: object | None = None
) -> list[str]:
    """Normalize ranked IDs and reject duplicate documents."""

    normalized_ids = [str(document_id) for document_id in ranked_ids]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for document_id in normalized_ids:
        if document_id in seen:
            duplicates.add(document_id)
        seen.add(document_id)
    if duplicates:
        query_context = f" for query {query_id!r}" if query_id is not None else ""
        raise ValueError(
            "ranked document IDs must be unique"
            f"{query_context}; duplicate IDs: {sorted(duplicates)}"
        )
    return normalized_ids


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return k


def stable_rank_ids(
    scores: Sequence[float], document_ids: Sequence[object], k: int
) -> list[str]:
    """Return top IDs by score, breaking equal scores by ascending ID."""

    k = _validate_k(k)
    if len(scores) != len(document_ids):
        raise ValueError("scores and document_ids must have equal lengths")
    normalized_ids = _validate_ranked_document_ids(document_ids)

    def sort_key(index: int) -> tuple[float, str]:
        score = float(scores[index])
        if math.isnan(score):
            score = float("-inf")
        return (-score, normalized_ids[index])

    order = sorted(range(len(normalized_ids)), key=sort_key)
    return [normalized_ids[index] for index in order[:k]]


def _gain(relevance: float) -> float:
    return 2.0 ** float(relevance) - 1.0


def _qrel_lookup(qrels: pd.DataFrame) -> dict[str, dict[str, float]]:
    normalized = _normalized_qrels(qrels)
    lookup: dict[str, dict[str, float]] = {}
    for query_id, group in normalized.groupby("query_id", sort=False):
        query_lookup: dict[str, float] = {}
        for document_id, relevance in zip(group["document_id"], group["relevance"]):
            query_lookup[document_id] = max(
                query_lookup.get(document_id, 0.0), float(relevance)
            )
        lookup[query_id] = query_lookup
    return lookup


def ndcg_at_k(
    ranked_ids: Sequence[object], qrels_for_query: Mapping[Any, float], k: int = 5
) -> float:
    """Compute graded nDCG using gain ``2**relevance - 1``."""

    k = _validate_k(k)
    normalized_ranked_ids = _validate_ranked_document_ids(ranked_ids)
    normalized_qrels = {
        str(document_id): float(relevance)
        for document_id, relevance in qrels_for_query.items()
    }
    observed = sum(
        _gain(normalized_qrels.get(str(document_id), 0.0)) / math.log2(rank + 2)
        for rank, document_id in enumerate(normalized_ranked_ids[:k])
    )
    ideal_values = sorted(normalized_qrels.values(), reverse=True)[:k]
    ideal = sum(
        _gain(relevance) / math.log2(rank + 2)
        for rank, relevance in enumerate(ideal_values)
    )
    return float(observed / ideal) if ideal else 0.0


def evaluate_rankings(
    rankings: Mapping[Any, Sequence[object]], qrels: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    """Return per-query nDCG and retrieved-positive count at ``k``."""

    k = _validate_k(k)
    qrel_lookup = _qrel_lookup(qrels)
    rows: list[dict[str, object]] = []
    for raw_query_id, ranked_ids in rankings.items():
        query_id = str(raw_query_id)
        query_qrels = qrel_lookup.get(query_id, {})
        normalized_ranked_ids = _validate_ranked_document_ids(ranked_ids, query_id)
        top_ids = normalized_ranked_ids[:k]
        rows.append(
            {
                "query_id": query_id,
                f"ndcg@{k}": ndcg_at_k(top_ids, query_qrels, k),
                f"retrieved_positive@{k}": sum(
                    query_qrels.get(document_id, 0.0) > 0
                    for document_id in top_ids
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["query_id", f"ndcg@{k}", f"retrieved_positive@{k}"],
    )


def compute_grade_aware_recall(
    rankings: Mapping[Any, Sequence[object]],
    qrels: pd.DataFrame,
    k: int,
    minimum_relevance: float,
) -> float:
    """Macro-average retrieved relevance gain for qrels above a grade floor."""

    k = _validate_k(k)
    minimum_relevance = float(minimum_relevance)
    if not math.isfinite(minimum_relevance):
        raise ValueError("minimum_relevance must be finite")
    qrel_lookup = _qrel_lookup(qrels)
    normalized_rankings = {
        str(query_id): _validate_ranked_document_ids(ranked_ids, query_id)[:k]
        for query_id, ranked_ids in rankings.items()
    }

    recalls: list[float] = []
    for query_id, query_qrels in qrel_lookup.items():
        eligible = {
            document_id: relevance
            for document_id, relevance in query_qrels.items()
            if relevance >= minimum_relevance and _gain(relevance) > 0
        }
        if not eligible:
            continue
        total_gain = sum(_gain(relevance) for relevance in eligible.values())
        retrieved_ids = set(normalized_rankings.get(query_id, ()))
        retrieved_gain = sum(
            _gain(eligible[document_id])
            for document_id in retrieved_ids
            if document_id in eligible
        )
        recalls.append(min(1.0, retrieved_gain / total_gain))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return item_method()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_run_manifest(
    output_path: Path, manifest: dict[str, object]
) -> None:
    """Write deterministic JSON to an explicit, previously unused path."""

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output_path}")
    payload = json.dumps(
        manifest,
        default=_json_default,
        indent=2,
        sort_keys=True,
    )
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")


def write_submission(
    rankings: Mapping[Any, Sequence[object]],
    test_queries: pd.DataFrame,
    documents: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Write a valid five-document-per-query competition submission.

    The helper preserves test-query order and ranking order, validates exact
    query coverage, unique top-five IDs, and corpus membership, and never
    creates parent directories or overwrites an existing file.
    """

    _require_columns(test_queries, "test_queries", _TEST_QUERY_COLUMNS)
    _require_columns(documents, "documents", {"document_id"})
    query_ids = test_queries["query_id"]
    if query_ids.isna().any():
        raise ValueError("test_queries query_id values must not be missing")
    normalized_query_ids = [str(query_id) for query_id in query_ids]
    if len(set(normalized_query_ids)) != len(normalized_query_ids):
        raise ValueError("test_queries query_id values must be unique")

    document_ids = documents["document_id"]
    if document_ids.isna().any():
        raise ValueError("documents document_id values must not be missing")
    normalized_document_ids = [str(document_id) for document_id in document_ids]
    if len(set(normalized_document_ids)) != len(normalized_document_ids):
        raise ValueError("documents document_id values must be unique")
    corpus_ids = set(normalized_document_ids)

    normalized_rankings: dict[str, list[object]] = {}
    for raw_query_id, ranked_ids in rankings.items():
        query_id = str(raw_query_id)
        if query_id in normalized_rankings:
            raise ValueError(f"rankings contains duplicate query ID: {query_id}")
        normalized_rankings[query_id] = list(ranked_ids)
    expected_query_ids = set(normalized_query_ids)
    ranking_query_ids = set(normalized_rankings)
    missing_query_ids = expected_query_ids - ranking_query_ids
    extra_query_ids = ranking_query_ids - expected_query_ids
    if missing_query_ids or extra_query_ids:
        raise ValueError(
            "rankings query coverage mismatch; "
            f"missing={sorted(missing_query_ids)}, extra={sorted(extra_query_ids)}"
        )

    rows: list[dict[str, str]] = []
    for query_id in normalized_query_ids:
        top_ids = [str(document_id) for document_id in normalized_rankings[query_id][:5]]
        if len(top_ids) != 5:
            raise ValueError(f"ranking for query {query_id} must contain at least five IDs")
        if len(set(top_ids)) != 5:
            raise ValueError(f"ranking for query {query_id} has duplicate top-five IDs")
        unknown_ids = set(top_ids) - corpus_ids
        if unknown_ids:
            raise ValueError(
                f"ranking for query {query_id} contains unknown document IDs: "
                f"{sorted(unknown_ids)}"
            )
        rows.extend(
            {"QueryId": query_id, "DocumentId": document_id}
            for document_id in top_ids
        )

    submission = pd.DataFrame(rows, columns=["QueryId", "DocumentId"])
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        submission.to_csv(handle, index=False)
    return submission

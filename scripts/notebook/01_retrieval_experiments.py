# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Agricultural Extension Retrieval Experiments
#
# Notebook source of truth for sparse, dense, and gated reranking experiments.
# Run sparse profile locally before enabling cloud-only profiles.

# %%
from __future__ import annotations

import hashlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    source_file = globals().get("__file__")
    candidates: list[Path] = []
    if source_file:
        candidates.extend(Path(source_file).resolve().parents)
    candidates.extend((cwd, *cwd.parents))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "scripts").is_dir()
        ):
            return candidate
    return cwd


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
SUBMISSIONS_DIR = PROJECT_ROOT / "scripts" / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
PROFILE = os.environ.get("AGRO_RAG_PROFILE", "sparse_local")
SUPPORTED_PROFILES = {
    "sparse_local",
    "dense_cloud",
    "rerank_local",
    "rerank_cloud",
    "cross_encoder_cloud",
}
if PROFILE not in SUPPORTED_PROFILES:
    supported_profiles = ", ".join(sorted(SUPPORTED_PROFILES))
    raise RuntimeError(
        f"Unsupported AGRO_RAG_PROFILE={PROFILE!r}. "
        f"Expected one of: {supported_profiles}."
    )
DENSE_PROFILES = {
    "dense_cloud",
    "rerank_local",
    "rerank_cloud",
    "cross_encoder_cloud",
}
RERANK_PROFILES = {
    "rerank_local",
    "rerank_cloud",
    "cross_encoder_cloud",
}
LAMBDAMART_PROFILES = {"rerank_local", "rerank_cloud"}
CROSS_ENCODER_PROFILE = "cross_encoder_cloud"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL_CARD_URL = (
    f"https://huggingface.co/{CROSS_ENCODER_MODEL_NAME}"
)
CROSS_ENCODER_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
LIGHTGBM_LICENSE_URL = (
    "https://github.com/microsoft/LightGBM/blob/master/LICENSE"
)
RERANK_CANDIDATE_K = 100
RERANK_GATE_MIN_RECALL = 0.80
RERANK_GATE_MIN_MARGIN_OVER_TOP5 = 0.20


def _rerank_artifact_paths(profile: str = PROFILE) -> tuple[Path, ...]:
    artifact_paths = [
        OUTPUT_DIR / "reranker_experiments.csv",
        OUTPUT_DIR / "rerank_candidate_gate.csv",
        OUTPUT_DIR / "rerank_document_embeddings.npy",
        OUTPUT_DIR / "rerank_query_embeddings.npy",
    ]
    if profile in LAMBDAMART_PROFILES:
        artifact_paths.append(SUBMISSIONS_DIR / "submission-4-lambdamart.csv")
    if profile == CROSS_ENCODER_PROFILE or os.environ.get(
        "AGRO_RAG_RUN_CROSS_ENCODER"
    ) == "1":
        artifact_paths.append(SUBMISSIONS_DIR / "submission-5-cross-encoder.csv")
    return tuple(artifact_paths)


def _invalidate_rerank_artifacts() -> list[str]:
    """Remove outputs whose provenance must be regenerated for selected runs."""
    removed: list[str] = []
    for path in _rerank_artifact_paths(PROFILE):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    if removed:
        print({"invalidated_rerank_artifacts": removed})
    return removed


if PROFILE in RERANK_PROFILES:
    # Clear stale artifacts before optional imports, data resolution, or sparse
    # execution can fail and leave prior rerank outputs looking current.
    _invalidate_rerank_artifacts()


if PROFILE in DENSE_PROFILES:
    dense_model_override = os.environ.get("AGRO_RAG_DENSE_MODEL")
    if dense_model_override is None:
        DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        DENSE_MODEL_LICENSE_URL = (
            "https://www.apache.org/licenses/LICENSE-2.0"
        )
        DENSE_MODEL_LICENSE_DESCRIPTION = "Apache-2.0"
    else:
        DENSE_MODEL_NAME = dense_model_override
        DENSE_MODEL_LICENSE_URL = os.environ.get(
            "AGRO_RAG_DENSE_LICENSE_URL"
        )
        if not DENSE_MODEL_LICENSE_URL:
            raise RuntimeError(
                "AGRO_RAG_DENSE_LICENSE_URL is required when "
                "AGRO_RAG_DENSE_MODEL overrides the disclosed default."
            )
        DENSE_MODEL_LICENSE_DESCRIPTION = (
            "custom license declared by AGRO_RAG_DENSE_LICENSE_URL"
        )
    DENSE_MODEL_CARD_URL = f"https://huggingface.co/{DENSE_MODEL_NAME}"
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            f"PROFILE='{PROFILE}' requires importable Torch and "
            "sentence-transformers; install the dense extra in cloud "
            "execution."
        ) from exc
else:
    DENSE_MODEL_NAME = None
    DENSE_MODEL_CARD_URL = None
    DENSE_MODEL_LICENSE_URL = None
    DENSE_MODEL_LICENSE_DESCRIPTION = None
    SentenceTransformer = None
    torch = None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if PROFILE not in DENSE_PROFILES:
        return
    if torch is None:
        raise RuntimeError(
            f"PROFILE='{PROFILE}' cannot seed inference because Torch "
            "is unavailable."
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

# %% [markdown]
# ## Competition data contract

# %%
DATA_FILENAMES = {
    "documents": "documents.csv",
    "train_queries": "train_queries.csv",
    "qrels": "qrels_train.csv",
    "test_queries": "test_queries.csv",
    "sample_submission": "sample_submission.csv",
}


def resolve_data_dir() -> Path:
    configured = os.environ.get("AGRO_RAG_DATA_DIR")
    if configured:
        data_dir = Path(configured).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"AGRO_RAG_DATA_DIR does not exist: {data_dir}")
        if not data_dir.is_dir():
            raise NotADirectoryError(
                f"AGRO_RAG_DATA_DIR is not a directory: {data_dir}"
            )
        return data_dir

    local_data_dir = (PROJECT_ROOT / "data").resolve()
    if local_data_dir.is_dir():
        return local_data_dir

    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            "KaggleHub is unavailable and AGRO_RAG_DATA_DIR is not configured."
        ) from exc
    try:
        return Path(
            kagglehub.competition_download(
                "agricultural-extension-rag-smart-retrieval-for-farmers"
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            "KaggleHub could not download the competition data; set "
            "AGRO_RAG_DATA_DIR to a local competition-data directory."
        ) from exc


def load_competition_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        name: pd.read_csv(data_dir / filename)
        for name, filename in DATA_FILENAMES.items()
    }
    for name in ("documents", "train_queries", "qrels", "test_queries"):
        for column in ("query_id", "document_id"):
            if column in tables[name]:
                tables[name][column] = tables[name][column].astype(str)
    tables["sample_submission"]["QueryId"] = tables["sample_submission"][
        "QueryId"
    ].astype(str)
    tables["sample_submission"]["DocumentId"] = tables["sample_submission"][
        "DocumentId"
    ].astype(str)
    tables["documents"]["document_id"] = tables["documents"][
        "document_id"
    ].astype(str)
    return tables


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    documents = tables["documents"]
    train_queries = tables["train_queries"]
    qrels = tables["qrels"]
    test_queries = tables["test_queries"]
    required = {
        "documents": {"document_id", "title", "text"},
        "train_queries": {"query_id", "query"},
        "qrels": {"query_id", "document_id", "relevance"},
        "test_queries": {"query_id", "query"},
    }
    for name, columns in required.items():
        missing = columns - set(tables[name].columns)
        assert not missing, f"{name} missing columns: {sorted(missing)}"
    assert documents["document_id"].is_unique
    assert train_queries["query_id"].is_unique
    assert test_queries["query_id"].is_unique
    assert qrels["relevance"].between(0, 3).all()
    assert set(qrels["document_id"]).issubset(set(documents["document_id"]))
    assert set(qrels["query_id"]).issubset(set(train_queries["query_id"]))

# %% [markdown]
# ## Deterministic grouped split

# %%
def build_grouped_split(
    train_queries: pd.DataFrame,
    qrels: pd.DataFrame,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[set[str], set[str]]:
    positive = qrels[qrels["relevance"] > 0]
    family = positive.groupby("query_id")["document_id"].apply(
        lambda values: "|".join(sorted(set(values)))
    )
    groups = train_queries["query_id"].map(family).fillna(
        train_queries["query_id"].map(lambda qid: f"singleton:{qid}")
    )
    group_ids = sorted(set(groups))
    ordered = sorted(
        group_ids,
        key=lambda group: hashlib.md5(
            f"{seed}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    n_validation = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * validation_fraction)),
    )
    validation_groups = set(ordered[-n_validation:])
    group_by_query = dict(zip(train_queries["query_id"], groups))
    validation_ids = {
        query_id for query_id, group in group_by_query.items()
        if group in validation_groups
    }
    train_ids = set(train_queries["query_id"]) - validation_ids
    assert train_ids and validation_ids
    assert train_ids.isdisjoint(validation_ids)
    return train_ids, validation_ids

# %% [markdown]
# ## Retrieval metrics

# %%
def ndcg_at_k(
    ranked_ids: list[str],
    qrels_for_query: dict[str, float],
    k: int = 5,
) -> float:
    observed = sum(
        (2.0 ** qrels_for_query.get(doc_id, 0.0) - 1.0)
        / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked_ids[:k])
    )
    ideal_values = sorted(qrels_for_query.values(), reverse=True)[:k]
    ideal = sum(
        (2.0 ** relevance - 1.0) / math.log2(rank + 2)
        for rank, relevance in enumerate(ideal_values)
    )
    return observed / ideal if ideal else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]], qrels: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    ndcg_column = f"ndcg@{k}"
    retrieved_positive_column = f"retrieved_positive@{k}"
    grouped = {
        str(query_id): dict(zip(group["document_id"], group["relevance"]))
        for query_id, group in qrels.groupby("query_id")
    }
    rows = []
    for query_id, ranked_ids in rankings.items():
        rows.append({
            "query_id": str(query_id),
            ndcg_column: ndcg_at_k(ranked_ids, grouped.get(str(query_id), {}), k),
            retrieved_positive_column: sum(
                grouped.get(str(query_id), {}).get(doc_id, 0) > 0
                for doc_id in ranked_ids[:k]
            ),
        })
    return pd.DataFrame(rows)


def mean_ndcg_at_k(metrics: pd.DataFrame, k: int = 5) -> float:
    return float(metrics[f"ndcg@{k}"].mean())

# %% [markdown]
# ## Metric contract smoke tests
#
# These assertions define the expected metric behavior before implementation.

# %%
assert ndcg_at_k(["d1", "d2"], {"d1": 3, "d2": 0}) == 1.0
assert ndcg_at_k(["d2", "d1"], {"d1": 3, "d2": 0}) < 1.0
toy_metrics = pd.DataFrame({"ndcg@5": [1.0, 0.5]})
mean_ndcg_at_5 = mean_ndcg_at_k(toy_metrics)
assert mean_ndcg_at_5 == 0.75

# %% [markdown]
# ## Competition contract execution

# %%
tables = load_competition_files(resolve_data_dir())
validate_tables(tables)

positive = tables["qrels"][tables["qrels"]["relevance"] > 0]
family = positive.groupby("query_id")["document_id"].apply(
    lambda values: "|".join(sorted(set(values)))
)
groups = tables["train_queries"]["query_id"].map(family).fillna(
    tables["train_queries"]["query_id"].map(
        lambda qid: f"singleton:{qid}"
    )
)
group_count = len(set(groups))
if group_count < 5:
    raise RuntimeError(
        "Grouped split requires at least five query groups; "
        f"found {group_count}."
    )

train_query_ids, validation_query_ids = build_grouped_split(
    tables["train_queries"], tables["qrels"], seed=SEED
)
if len(validation_query_ids) < 0.1 * len(tables["train_queries"]):
    raise RuntimeError(
        "Grouped split validation queries must be at least 10% of training "
        f"queries; found {len(validation_query_ids)} of "
        f"{len(tables['train_queries'])}."
    )

print({
    "documents": len(tables["documents"]),
    "train_queries": len(train_query_ids),
    "validation_queries": len(validation_query_ids),
    "total_train_queries": len(tables["train_queries"]),
    "qrels": len(tables["qrels"]),
    "test_queries": len(tables["test_queries"]),
    "query_groups": group_count,
})

# %% [markdown]
# ## Local sparse retrieval
#
# These lexical lanes follow the robust sparse-baseline evaluation practice
# described by BEIR (Thakur et al., arXiv:2104.08663). The local profile uses
# no pretrained model or external corpus.

# %%
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def document_text(documents: pd.DataFrame, title_weight: int = 1) -> list[str]:
    titles = documents["title"].fillna("").astype(str)
    bodies = documents["text"].fillna("").astype(str)
    return [
        " ".join(([title] * title_weight) + [body]).strip()
        for title, body in zip(titles, bodies)
    ]


def top_document_ids(
    scores: np.ndarray, document_ids: list[str], k: int
) -> list[str]:
    ids = np.asarray(document_ids, dtype=str)
    order = np.lexsort((ids, -np.asarray(scores, dtype=float)))
    return ids[order[:k]].tolist()


def _rankings_with_metadata(
    scores: np.ndarray,
    query_ids: pd.Series,
    document_ids: pd.Series,
    k: int,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    score_array = np.asarray(scores, dtype=float)
    document_id_list = document_ids.astype(str).tolist()
    document_id_array = np.asarray(document_id_list, dtype=str)
    rankings: dict[str, list[str]] = {}
    scores_by_query: dict[str, dict[str, float]] = {}
    ranks_by_query: dict[str, dict[str, float]] = {}
    for query_id, score_row in zip(query_ids, score_array):
        query_key = str(query_id)
        order = np.lexsort((document_id_array, -score_row))
        rankings[query_key] = document_id_array[order[:k]].tolist()
        scores_by_query[query_key] = {
            document_id: float(score)
            for document_id, score in zip(document_id_list, score_row)
        }
        ranks_by_query[query_key] = {
            document_id_array[index].item(): float(rank)
            for rank, index in enumerate(order)
        }
    return rankings, scores_by_query, ranks_by_query


def encode_documents_and_queries(
    model_name: str,
    documents: pd.DataFrame,
    queries: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode corpus and query text with the explicitly selected cloud model."""
    if PROFILE not in DENSE_PROFILES or SentenceTransformer is None:
        raise RuntimeError(
            "Dense encoding is opt-in; set AGRO_RAG_PROFILE to a dense "
            "profile."
        )
    model = SentenceTransformer(model_name)
    document_texts = document_text(documents)
    query_texts = queries["query"].fillna("").astype(str).tolist()
    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(document_embeddings), np.asarray(query_embeddings)


def retrieve_dense(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: pd.Series,
    document_ids: pd.Series,
    k: int = 5,
) -> dict[str, list[str]]:
    """Retrieve documents by normalized embedding dot product."""
    return retrieve_dense_with_metadata(
        query_embeddings,
        document_embeddings,
        query_ids,
        document_ids,
        k=k,
    )[0]


def retrieve_dense_with_metadata(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: pd.Series,
    document_ids: pd.Series,
    k: int = 5,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Return dense rankings plus cosine scores and zero-based ranks."""
    query_embeddings = np.asarray(query_embeddings)
    document_embeddings = np.asarray(document_embeddings)
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("query and document embeddings must be two-dimensional")
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError("query and document embedding dimensions must match")
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("query_ids must match number of query embeddings")
    if len(document_ids) != document_embeddings.shape[0]:
        raise ValueError("document_ids must match number of document embeddings")
    scores = query_embeddings @ document_embeddings.T
    return _rankings_with_metadata(scores, query_ids, document_ids, k)


TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def retrieve_tfidf_with_metadata(
    queries: pd.DataFrame,
    documents: pd.DataFrame,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    title_weight: int = 1,
    k: int = 5,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    texts = document_text(documents, title_weight)
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    doc_matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform(queries["query"].fillna("").astype(str))
    scores = cosine_similarity(query_matrix, doc_matrix)
    return _rankings_with_metadata(
        scores,
        queries["query_id"],
        documents["document_id"],
        k,
    )


def retrieve_tfidf(
    queries: pd.DataFrame,
    documents: pd.DataFrame,
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    title_weight: int = 1,
    k: int = 5,
) -> dict[str, list[str]]:
    return retrieve_tfidf_with_metadata(
        queries,
        documents,
        analyzer=analyzer,
        ngram_range=ngram_range,
        title_weight=title_weight,
        k=k,
    )[0]


def validate_submission(
    submission: pd.DataFrame,
    test_queries: pd.DataFrame,
    documents: pd.DataFrame,
) -> None:
    """Assert competition format, coverage, uniqueness, and row ordering."""
    assert list(submission.columns) == ["QueryId", "DocumentId"]

    expected_query_ids = test_queries["query_id"].astype(str).tolist()
    expected_queries = set(expected_query_ids)
    assert len(expected_query_ids) == 200
    assert len(expected_queries) == 200

    query_ids = submission["QueryId"].astype(str)
    document_ids = submission["DocumentId"].astype(str)
    assert set(query_ids) == expected_queries
    assert len(submission) == 1_000

    expected_query_order = [
        query_id for query_id in expected_query_ids for _ in range(5)
    ]
    assert query_ids.tolist() == expected_query_order, (
        "submission must preserve test-query and per-query ranking order"
    )

    normalized = submission.assign(QueryId=query_ids, DocumentId=document_ids)
    grouped = normalized.groupby("QueryId", sort=False)["DocumentId"].agg(
        ["size", "nunique"]
    )
    assert grouped["size"].eq(5).all()
    assert grouped["nunique"].eq(5).all()
    assert set(document_ids).issubset(
        set(documents["document_id"].astype(str))
    )


def write_submission(
    rankings: dict[str, list[str]],
    test_queries: pd.DataFrame,
    documents: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Write top-five rankings without changing retrieval order."""
    rows = [
        {"QueryId": str(query_id), "DocumentId": str(document_id)}
        for query_id in test_queries["query_id"].astype(str)
        for document_id in rankings[str(query_id)][:5]
    ]
    submission = pd.DataFrame(rows, columns=["QueryId", "DocumentId"])
    validate_submission(submission, test_queries, documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def retrieve_bm25(
    queries: pd.DataFrame,
    documents: pd.DataFrame,
    k1: float = 1.5,
    b: float = 0.75,
    k: int = 5,
) -> dict[str, list[str]]:
    doc_ids = documents["document_id"].astype(str).tolist()
    tokenized = [_tokens(text) for text in document_text(documents)]
    lengths = np.asarray([len(tokens) for tokens in tokenized], dtype=float)
    average_length = max(float(lengths.mean()), 1.0)
    term_document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        term_document_frequency.update(set(tokens))
    n_documents = len(tokenized)
    outputs: dict[str, list[str]] = {}
    for query_id, query in zip(queries["query_id"], queries["query"]):
        query_counts = Counter(_tokens(str(query)))
        scores = np.zeros(n_documents, dtype=float)
        for term, query_frequency in query_counts.items():
            document_frequency = term_document_frequency.get(term, 0)
            if not document_frequency:
                continue
            inverse_frequency = math.log1p(
                (n_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, tokens in enumerate(tokenized):
                term_frequency = tokens.count(term)
                if not term_frequency:
                    continue
                denominator = term_frequency + k1 * (
                    1.0 - b + b * lengths[index] / average_length
                )
                scores[index] += inverse_frequency * (
                    term_frequency * (k1 + 1.0) / denominator
                ) * query_frequency
        outputs[str(query_id)] = top_document_ids(scores, doc_ids, k)
    return outputs


def rrf_fuse(
    rank_lists: list[list[str]], rrf_k: int = 60, k: int = 5
) -> list[str]:
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, document_id in enumerate(rank_list):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
    return [
        document_id
        for document_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:k]
    ]


def fuse_rankings(
    ranking_sets: list[dict[str, list[str]]], k: int = 5
) -> dict[str, list[str]]:
    query_ids = ranking_sets[0].keys()
    return {
        query_id: rrf_fuse([ranking[query_id] for ranking in ranking_sets], k=k)
        for query_id in query_ids
    }


assert document_text(pd.DataFrame({
    "title": ["Title"], "text": ["body"],
}), title_weight=2) == ["Title Title body"]
assert top_document_ids(
    np.asarray([1.0, 1.0, 0.5]), ["d2", "d1", "d3"], 3
) == ["d1", "d2", "d3"]
assert rrf_fuse([["d2", "d1"], ["d1", "d3"]], rrf_k=60, k=3) == [
    "d1", "d2", "d3"
]
_metadata_test_documents = pd.DataFrame({
    "document_id": ["d2", "d1"],
    "title": ["Alpha", "Alpha"],
    "text": ["shared body", "shared body"],
})
_metadata_test_queries = pd.DataFrame({
    "query_id": ["q1"],
    "query": ["shared"],
})
_metadata_test_rankings, _metadata_test_scores, _metadata_test_ranks = (
    retrieve_tfidf_with_metadata(
        _metadata_test_queries,
        _metadata_test_documents,
        analyzer="word",
        ngram_range=(1, 1),
        k=2,
    )
)
assert _metadata_test_rankings["q1"] == ["d1", "d2"]
assert set(_metadata_test_scores["q1"]) == {"d1", "d2"}
assert _metadata_test_ranks["q1"]["d1"] == 0.0
_metadata_test_dense = retrieve_dense_with_metadata(
    np.asarray([[1.0, 0.0]]),
    np.asarray([[1.0, 0.0], [1.0, 0.0]]),
    pd.Series(["q1"]),
    pd.Series(["d2", "d1"]),
    k=2,
)
assert _metadata_test_dense[0]["q1"] == ["d1", "d2"]
assert _metadata_test_dense[1]["q1"]["d1"] == 1.0

# %% [markdown]
# ## Grouped-validation sparse experiment rows

# %%
validation_queries = tables["train_queries"].loc[
    tables["train_queries"]["query_id"].isin(validation_query_ids)
].copy()
validation_qrels = tables["qrels"].loc[
    tables["qrels"]["query_id"].isin(validation_query_ids)
].copy()


def positive_recall_at_k(
    rankings: dict[str, list[str]], qrels: pd.DataFrame, k: int
) -> float:
    positive = {
        str(query_id): set(
            group.loc[group["relevance"] > 0, "document_id"].astype(str)
        )
        for query_id, group in qrels.groupby("query_id")
    }
    recalls = []
    for query_id, relevant_ids in positive.items():
        if relevant_ids:
            recalls.append(
                len(set(rankings.get(query_id, [])[:k]) & relevant_ids)
                / len(relevant_ids)
            )
    return float(np.mean(recalls)) if recalls else 0.0


def _mean_positive_recall_at_k(
    rankings: dict[str, list[str]], qrels: pd.DataFrame, k: int = 5
) -> float:
    return positive_recall_at_k(rankings, qrels, k)


FEATURE_NAMES = [
    "sparse_score",
    "dense_score",
    "sparse_rank",
    "dense_rank",
    "rrf_rank",
    "query_token_overlap",
    "title_token_overlap",
    "crop_exact_match",
    "document_text_length",
]


def _word_set(text: str) -> set[str]:
    return set(_tokens(str(text)))


def build_candidate_features(
    query_rows: pd.DataFrame,
    documents: pd.DataFrame,
    rankings: dict[str, list[str]],
    retrieval_metadata: dict[str, dict[str, dict[str, float]]],
    qrels: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build leakage-safe candidate features and optional relevance labels."""
    document_frame = documents.copy()
    document_frame["document_id"] = document_frame["document_id"].astype(str)
    document_lookup = document_frame.set_index("document_id")

    sparse_scores = retrieval_metadata.get("sparse_score", {})
    dense_scores = retrieval_metadata.get("dense_score", {})
    sparse_ranks = retrieval_metadata.get("sparse_rank", {})
    dense_ranks = retrieval_metadata.get("dense_rank", {})
    qrel_lookup = None
    if qrels is not None:
        qrel_frame = qrels.copy()
        qrel_frame["query_id"] = qrel_frame["query_id"].astype(str)
        qrel_frame["document_id"] = qrel_frame["document_id"].astype(str)
        qrel_lookup = qrel_frame.set_index(
            ["query_id", "document_id"]
        )["relevance"]

    rows: list[dict[str, object]] = []
    labels: list[int] = []
    for _, query_row in query_rows.iterrows():
        query_id = str(query_row["query_id"])
        query_tokens = _word_set(query_row["query"])
        query_rankings = rankings.get(query_id, [])
        query_sparse_scores = sparse_scores.get(query_id, {})
        query_dense_scores = dense_scores.get(query_id, {})
        query_sparse_ranks = sparse_ranks.get(query_id, {})
        query_dense_ranks = dense_ranks.get(query_id, {})
        for rrf_rank, raw_document_id in enumerate(query_rankings):
            document_id = str(raw_document_id)
            document = document_lookup.loc[document_id]
            title_tokens = _word_set(document["title"])
            body_tokens = _word_set(document["text"])
            rows.append({
                "query_id": query_id,
                "document_id": document_id,
                "rrf_rank": rrf_rank,
                "sparse_score": float(
                    query_sparse_scores.get(document_id, 0.0)
                ),
                "dense_score": float(
                    query_dense_scores.get(document_id, 0.0)
                ),
                "sparse_rank": int(
                    query_sparse_ranks.get(document_id, 9999)
                ),
                "dense_rank": int(
                    query_dense_ranks.get(document_id, 9999)
                ),
                "query_token_overlap": len(
                    query_tokens & (title_tokens | body_tokens)
                ),
                "title_token_overlap": len(query_tokens & title_tokens),
                "crop_exact_match": int(
                    bool(query_tokens & _word_set(document.get("crop", "")))
                ),
                "document_text_length": len(str(document["text"])),
            })
            if qrel_lookup is None:
                labels.append(0)
            else:
                labels.append(int(qrel_lookup.get((query_id, document_id), 0)))
    features = pd.DataFrame(
        rows,
        columns=["query_id", "document_id", *FEATURE_NAMES],
    )
    return features, np.asarray(labels, dtype=np.int32)


# Targeted feature-contract assertions: labels are optional, metadata is not
# derived from qrels, candidate order follows query order, and absent metadata
# uses the documented defaults.
_feature_test_queries = pd.DataFrame({
    "query_id": ["q2", "q1"],
    "query": ["wheat advice", "maize advice"],
})
_feature_test_documents = pd.DataFrame({
    "document_id": ["d1", "d2"],
    "title": ["Maize advice", "Wheat advice"],
    "text": ["Maize body", "Wheat body"],
    "crop": ["maize", "wheat"],
})
_feature_test_rankings = {"q2": ["d2", "d1"], "q1": ["d1", "d2"]}
_feature_test_metadata = {
    "sparse_score": {"q1": {"d1": 0.7}},
    "dense_score": {"q1": {"d1": 0.9}},
    "sparse_rank": {"q1": {"d1": 0.0}},
    "dense_rank": {"q1": {"d1": 0.0}},
}
_feature_test_qrels = pd.DataFrame({
    "query_id": ["q2", "q1"],
    "document_id": ["d2", "d1"],
    "relevance": [1, 3],
})
_feature_test_frame, _feature_test_labels = build_candidate_features(
    _feature_test_queries,
    _feature_test_documents,
    _feature_test_rankings,
    _feature_test_metadata,
    _feature_test_qrels,
)
assert list(_feature_test_frame.columns) == [
    "query_id", "document_id", *FEATURE_NAMES
]
assert _feature_test_frame["query_id"].tolist() == ["q2", "q2", "q1", "q1"]
assert _feature_test_labels.tolist() == [1, 0, 3, 0]
assert _feature_test_frame.iloc[0]["sparse_score"] == 0.0
assert _feature_test_frame.iloc[0]["sparse_rank"] == 9999
_, _feature_test_default_labels = build_candidate_features(
    _feature_test_queries,
    _feature_test_documents,
    _feature_test_rankings,
    _feature_test_metadata,
)
assert _feature_test_default_labels.tolist() == [0, 0, 0, 0]


def rerank_with_lambdamart(
    train_features: pd.DataFrame,
    labels: np.ndarray,
    query_groups: list[int],
    validation_features: pd.DataFrame,
) -> np.ndarray:
    """Fit archive-style LambdaMART on supplied training groups only."""
    if len(labels) != len(train_features):
        raise ValueError("labels must align one-to-one with train_features")
    if sum(query_groups) != len(train_features):
        raise ValueError(
            "sum(query_groups) must equal number of training candidates"
        )
    try:
        from lightgbm import LGBMRanker
    except Exception as exc:
        raise RuntimeError(
            "LambdaMART requires importable LightGBM; install the rerank "
            "extra in the selected execution environment."
        ) from exc
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=SEED,
        verbosity=-1,
    )
    try:
        model.fit(
            train_features[FEATURE_NAMES],
            labels.astype(int),
            group=query_groups,
            eval_at=[5],
        )
        predictions = model.predict(validation_features[FEATURE_NAMES])
    except Exception as exc:
        raise RuntimeError("LambdaMART fitting or prediction failed.") from exc
    return np.asarray(predictions, dtype=float)


def _rank_candidates_by_score(
    candidate_features: pd.DataFrame,
    predictions: np.ndarray,
    k: int = 5,
) -> dict[str, list[str]]:
    predictions = np.asarray(predictions, dtype=float)
    if len(predictions) != len(candidate_features):
        raise ValueError("predictions must align one-to-one with candidates")
    scored = candidate_features[["query_id", "document_id"]].copy()
    scored["prediction"] = predictions
    outputs: dict[str, list[str]] = {}
    for query_id, group in scored.groupby("query_id", sort=False):
        ordered = group.sort_values(
            ["prediction", "document_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        outputs[str(query_id)] = ordered["document_id"].astype(str).head(k).tolist()
    return outputs


def _cross_encoder_enabled() -> bool:
    return (
        PROFILE == CROSS_ENCODER_PROFILE
        or (
            PROFILE in RERANK_PROFILES
            and os.environ.get("AGRO_RAG_RUN_CROSS_ENCODER") == "1"
        )
    )


def _lambdamart_enabled() -> bool:
    return PROFILE in LAMBDAMART_PROFILES


def rerank_with_cross_encoder(
    model_name: str,
    queries: pd.DataFrame,
    documents: pd.DataFrame,
    candidates: dict[str, list[str]],
    k: int = 5,
) -> dict[str, list[str]]:
    """Score candidate pairs without training and sort ties by document ID."""
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError(
            "Cross-encoder profile requires importable sentence-transformers "
            "and Torch; install the cross_encoder extra in cloud execution."
        ) from exc

    document_frame = documents.copy()
    document_frame["document_id"] = document_frame["document_id"].astype(str)
    document_lookup = document_frame.set_index("document_id")
    query_frame = queries.copy()
    query_frame["query_id"] = query_frame["query_id"].astype(str)
    query_lookup = query_frame.set_index("query_id")
    try:
        model = CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Cross-encoder model '{model_name}' could not be loaded."
        ) from exc

    outputs: dict[str, list[str]] = {}
    for raw_query_id, raw_candidate_ids in candidates.items():
        query_id = str(raw_query_id)
        if query_id not in query_lookup.index:
            raise ValueError(f"Missing query text for candidate query {query_id}")
        candidate_ids = [str(document_id) for document_id in raw_candidate_ids]
        if not candidate_ids:
            outputs[query_id] = []
            continue
        query_text = str(query_lookup.loc[query_id, "query"])
        pairs = [
            (
                query_text,
                f"{document_lookup.loc[document_id, 'title']}. "
                f"{document_lookup.loc[document_id, 'text']}",
            )
            for document_id in candidate_ids
        ]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            raise RuntimeError(
                f"Cross-encoder prediction failed for query {query_id}."
            ) from exc
        if len(scores) != len(candidate_ids):
            raise RuntimeError(
                "Cross-encoder returned a score count different from candidates."
            )
        outputs[query_id] = [
            document_id
            for document_id, _ in sorted(
                zip(candidate_ids, scores),
                key=lambda item: (-float(item[1]), item[0]),
            )[:k]
        ]
    return outputs


def _check_rerank_dependencies() -> None:
    """Fail before dense encoding when selected reranking dependencies miss."""
    if _lambdamart_enabled():
        try:
            from lightgbm import LGBMRanker as _LGBMRanker
        except Exception as exc:
            raise RuntimeError(
                "LambdaMART profile requires importable LightGBM; install "
                "the rerank extra in the selected execution environment."
            ) from exc
        del _LGBMRanker
    if _cross_encoder_enabled():
        try:
            from sentence_transformers import CrossEncoder as _CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "Cross-encoder profile requires importable "
                "sentence-transformers and Torch; install the cross_encoder "
                "extra in cloud execution."
            ) from exc
        del _CrossEncoder


def _tfidf_vocabulary_size(
    documents: pd.DataFrame,
    analyzer: str,
    ngram_range: tuple[int, int],
    title_weight: int,
) -> int:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    vectorizer.fit(document_text(documents, title_weight))
    return len(vectorizer.vocabulary_)


def _bm25_vocabulary_size(documents: pd.DataFrame, title_weight: int) -> int:
    return len({
        token
        for text in document_text(documents, title_weight)
        for token in _tokens(text)
    })


def _bm25_documents_with_title_weight(
    documents: pd.DataFrame, title_weight: int
) -> pd.DataFrame:
    weighted_documents = documents.copy()
    weighted_documents["title"] = [
        " ".join([title] * title_weight)
        for title in documents["title"].fillna("").astype(str)
    ]
    return weighted_documents


def _sparse_result_row(
    method: str,
    configuration: str,
    vocabulary_size: int | None,
    rankings: dict[str, list[str]],
    qrels: pd.DataFrame,
    runtime_seconds: float,
    decision: str,
    k: int = 5,
) -> dict[str, object]:
    expected_query_ids = set(qrels["query_id"].astype(str))
    assert set(rankings) == expected_query_ids, (
        f"{method} rankings must cover exactly validation queries; "
        f"expected {len(expected_query_ids)}, got {len(rankings)}"
    )
    metrics = evaluate_rankings(rankings, qrels, k=k)
    candidate_recall_at_k = _mean_positive_recall_at_k(rankings, qrels, k=k)
    return {
        "method": method,
        "profile": PROFILE,
        "split": "grouped_validation_80_20",
        "seed": SEED,
        "parameters": configuration,
        "vocabulary_size": vocabulary_size,
        "configuration": configuration,
        "mean_ndcg@5": mean_ndcg_at_k(metrics, k=k),
        "positive_recall@5": candidate_recall_at_k,
        "candidate_recall@5": candidate_recall_at_k,
        "runtime_seconds": runtime_seconds,
        "model_identifier": "none",
        "license": "not applicable; no pretrained model",
        "kaggle_status": "not submitted; validation only",
        "decision": decision,
        "interpretation": "Lexical validation run on fixed grouped holdout.",
    }


def run_sparse_experiments(
    validation_queries: pd.DataFrame,
    documents: pd.DataFrame,
    validation_qrels: pd.DataFrame,
    k: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    rankings_by_name: dict[str, dict[str, list[str]]] = {}
    rows: list[dict[str, object]] = []

    def run_one(
        name: str,
        method: str,
        configuration: str,
        vocabulary_size: int | None,
        decision: str,
        retrieve,
    ) -> None:
        started = time.perf_counter()
        rankings = retrieve()
        runtime_seconds = time.perf_counter() - started
        rankings_by_name[name] = rankings
        rows.append(_sparse_result_row(
            method,
            configuration,
            vocabulary_size,
            rankings,
            validation_qrels,
            runtime_seconds,
            decision,
            k=k,
        ))

    word_ngram_range = (1, 2)
    char_ngram_range = (3, 5)
    word_vocab_1 = _tfidf_vocabulary_size(
        documents, "word", word_ngram_range, title_weight=1
    )
    word_vocab_2 = _tfidf_vocabulary_size(
        documents, "word", word_ngram_range, title_weight=2
    )
    char_vocab_1 = _tfidf_vocabulary_size(
        documents, "char", char_ngram_range, title_weight=1
    )
    char_vocab_2 = _tfidf_vocabulary_size(
        documents, "char", char_ngram_range, title_weight=2
    )
    bm25_vocab_1 = _bm25_vocabulary_size(documents, title_weight=1)
    bm25_vocab_2 = _bm25_vocabulary_size(documents, title_weight=2)

    run_one(
        "word_tfidf_title_1",
        "TF-IDF word",
        "analyzer=word; ngram_range=(1, 2); title_weight=1",
        word_vocab_1,
        "not selected for first submission; lower nDCG@5 (0.526531) than "
        "selected character TF-IDF (0.543535)",
        lambda: retrieve_tfidf(
            validation_queries,
            documents,
            analyzer="word",
            ngram_range=word_ngram_range,
            title_weight=1,
            k=k,
        ),
    )
    run_one(
        "word_tfidf_title_2",
        "TF-IDF word",
        "analyzer=word; ngram_range=(1, 2); title_weight=2",
        word_vocab_2,
        "not selected for first submission; lower nDCG@5 (0.539542) than "
        "selected character TF-IDF (0.543535)",
        lambda: retrieve_tfidf(
            validation_queries,
            documents,
            analyzer="word",
            ngram_range=word_ngram_range,
            title_weight=2,
            k=k,
        ),
    )
    run_one(
        "char_tfidf_title_1",
        "TF-IDF character",
        "analyzer=char; ngram_range=(3, 5); title_weight=1",
        char_vocab_1,
        "not selected for first submission; lower nDCG@5 (0.532335) than "
        "selected character TF-IDF (0.543535)",
        lambda: retrieve_tfidf(
            validation_queries,
            documents,
            analyzer="char",
            ngram_range=char_ngram_range,
            title_weight=1,
            k=k,
        ),
    )
    run_one(
        "char_tfidf_title_2",
        "TF-IDF character",
        "analyzer=char; ngram_range=(3, 5); title_weight=2",
        char_vocab_2,
        "selected for first submission",
        lambda: retrieve_tfidf(
            validation_queries,
            documents,
            analyzer="char",
            ngram_range=char_ngram_range,
            title_weight=2,
            k=k,
        ),
    )
    run_one(
        "bm25_title_1",
        "BM25",
        "k1=1.5; b=0.75; title_weight=1",
        bm25_vocab_1,
        "not selected for first submission; lower nDCG@5 (0.432336) than "
        "selected character TF-IDF (0.543535)",
        lambda: retrieve_bm25(
            validation_queries,
            documents,
            k1=1.5,
            b=0.75,
            k=k,
        ),
    )
    run_one(
        "bm25_title_2",
        "BM25",
        "k1=1.5; b=0.75; title_weight=2",
        bm25_vocab_2,
        "not selected for first submission; lower nDCG@5 (0.452283) than "
        "selected character TF-IDF (0.543535)",
        lambda: retrieve_bm25(
            validation_queries,
            _bm25_documents_with_title_weight(documents, title_weight=2),
            k1=1.5,
            b=0.75,
            k=k,
        ),
    )

    rrf_configuration = (
        "rrf_k=60; k=5; components=word_tfidf_title_1+"
        "char_tfidf_title_1+bm25_title_1"
    )
    started = time.perf_counter()
    rrf_rankings = fuse_rankings([
        rankings_by_name["word_tfidf_title_1"],
        rankings_by_name["char_tfidf_title_1"],
        rankings_by_name["bm25_title_1"],
    ], k=k)
    runtime_seconds = time.perf_counter() - started
    rankings_by_name["sparse_rrf"] = rrf_rankings
    rows.append(_sparse_result_row(
        "Sparse RRF",
        rrf_configuration,
        None,
        rrf_rankings,
        validation_qrels,
        runtime_seconds,
        "not selected for first submission; lower nDCG@5 (0.499061) than "
        "selected character TF-IDF (0.543535)",
        k=k,
    ))

    results = pd.DataFrame(rows)
    return results, rankings_by_name


DENSE_CANDIDATE_K = 100


def _dense_decision(
    method: str, ndcg: float, sparse_ndcg: float
) -> str:
    if ndcg > sparse_ndcg:
        return (
            f"adopted; {method} improves grouped-validation nDCG@5 "
            f"({ndcg:.6f} > sparse char_tfidf_title_2 {sparse_ndcg:.6f})"
        )
    return (
        f"not adopted; {method} does not improve grouped-validation nDCG@5 "
        f"({ndcg:.6f} <= sparse char_tfidf_title_2 {sparse_ndcg:.6f}); "
        "retain sparse char_tfidf_title_2"
    )


def _dense_result_row(
    method: str,
    configuration: str,
    rankings: dict[str, list[str]],
    qrels: pd.DataFrame,
    runtime_seconds: float,
    encoding_runtime_seconds: float,
    embedding_dimension: int,
    decision: str,
    k: int = 5,
) -> dict[str, object]:
    expected_query_ids = set(qrels["query_id"].astype(str))
    assert set(rankings) == expected_query_ids, (
        f"{method} rankings must cover exactly validation queries; "
        f"expected {len(expected_query_ids)}, got {len(rankings)}"
    )
    metrics = evaluate_rankings(rankings, qrels, k=k)
    positive_recall = positive_recall_at_k(rankings, qrels, k=k)
    return {
        "method": method,
        "profile": PROFILE,
        "split": "grouped_validation_80_20",
        "seed": SEED,
        "parameters": configuration,
        "vocabulary_size": None,
        "configuration": configuration,
        "embedding_dimension": embedding_dimension,
        "mean_ndcg@5": mean_ndcg_at_k(metrics, k=k),
        "positive_recall@5": positive_recall,
        "candidate_recall@5": positive_recall,
        "candidate_recall@50": positive_recall_at_k(rankings, qrels, k=50),
        "candidate_recall@100": positive_recall_at_k(rankings, qrels, k=100),
        "runtime_seconds": runtime_seconds,
        "encoding_runtime_seconds": encoding_runtime_seconds,
        "model_identifier": DENSE_MODEL_NAME,
        "model_card_url": DENSE_MODEL_CARD_URL,
        "license_url": DENSE_MODEL_LICENSE_URL,
        "license": (
            f"{DENSE_MODEL_LICENSE_DESCRIPTION}; "
            f"{DENSE_MODEL_LICENSE_URL}"
        ),
        "kaggle_status": "validated test artifact generated; not submitted",
        "decision": decision,
        "interpretation": (
            "Dense validation uses the fixed grouped holdout and the disclosed "
            "normalized embedding model."
        ),
    }


def run_dense_experiments(
    validation_queries: pd.DataFrame,
    documents: pd.DataFrame,
    validation_qrels: pd.DataFrame,
    test_queries: pd.DataFrame,
    k: int = 5,
    candidate_k: int = DENSE_CANDIDATE_K,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    """Evaluate dense and sparse-best+dense RRF on one grouped holdout."""
    if PROFILE not in DENSE_PROFILES or DENSE_MODEL_NAME is None:
        raise RuntimeError(
            "Dense experiments require an opt-in dense profile."
        )
    candidate_k = max(candidate_k, 100)
    all_queries = pd.concat(
        [validation_queries, test_queries], ignore_index=True
    )

    encoding_started = time.perf_counter()
    document_embeddings, all_query_embeddings = encode_documents_and_queries(
        DENSE_MODEL_NAME,
        documents,
        all_queries,
    )
    encoding_runtime_seconds = time.perf_counter() - encoding_started
    assert document_embeddings.ndim == 2
    assert all_query_embeddings.ndim == 2
    assert len(document_embeddings) == len(documents)
    assert len(all_query_embeddings) == len(all_queries)
    assert document_embeddings.shape[1] == all_query_embeddings.shape[1]
    embedding_dimension = int(document_embeddings.shape[1])

    np.save(OUTPUT_DIR / "dense_document_embeddings.npy", document_embeddings)
    np.save(OUTPUT_DIR / "dense_query_embeddings.npy", all_query_embeddings)

    validation_count = len(validation_queries)
    validation_query_embeddings = all_query_embeddings[:validation_count]
    test_query_embeddings = all_query_embeddings[validation_count:]
    document_ids = documents["document_id"]

    dense_ranking_started = time.perf_counter()
    dense_validation_rankings = retrieve_dense(
        validation_query_embeddings,
        document_embeddings,
        validation_queries["query_id"],
        document_ids,
        k=candidate_k,
    )
    dense_test_rankings = retrieve_dense(
        test_query_embeddings,
        document_embeddings,
        test_queries["query_id"],
        document_ids,
        k=candidate_k,
    )
    dense_ranking_runtime_seconds = time.perf_counter() - dense_ranking_started
    dense_runtime_seconds = (
        encoding_runtime_seconds + dense_ranking_runtime_seconds
    )

    hybrid_started = time.perf_counter()
    sparse_validation_rankings = retrieve_tfidf(
        validation_queries,
        documents,
        analyzer="char",
        ngram_range=(3, 5),
        title_weight=2,
        k=candidate_k,
    )
    sparse_test_rankings = retrieve_tfidf(
        test_queries,
        documents,
        analyzer="char",
        ngram_range=(3, 5),
        title_weight=2,
        k=candidate_k,
    )
    hybrid_validation_rankings = fuse_rankings(
        [sparse_validation_rankings, dense_validation_rankings],
        k=candidate_k,
    )
    hybrid_test_rankings = fuse_rankings(
        [sparse_test_rankings, dense_test_rankings],
        k=candidate_k,
    )
    hybrid_runtime_seconds = (
        encoding_runtime_seconds
        + dense_ranking_runtime_seconds
        + (time.perf_counter() - hybrid_started)
    )

    sparse_ndcg = mean_ndcg_at_k(
        evaluate_rankings(sparse_validation_rankings, validation_qrels, k=k),
        k=k,
    )
    dense_ndcg = mean_ndcg_at_k(
        evaluate_rankings(dense_validation_rankings, validation_qrels, k=k),
        k=k,
    )
    hybrid_ndcg = mean_ndcg_at_k(
        evaluate_rankings(hybrid_validation_rankings, validation_qrels, k=k),
        k=k,
    )
    dense_decision = _dense_decision(
        "dense retrieval", dense_ndcg, sparse_ndcg
    )
    hybrid_decision = _dense_decision(
        "sparse+dense RRF", hybrid_ndcg, sparse_ndcg
    )

    dense_configuration = (
        f"model={DENSE_MODEL_NAME}; normalized_embeddings=True; "
        f"candidate_k={candidate_k}; k={k}"
    )
    hybrid_configuration = (
        f"components=char_tfidf_title_2+dense; rrf_k=60; "
        f"candidate_k={candidate_k}; k={k}; model={DENSE_MODEL_NAME}"
    )
    results = pd.DataFrame([
        _dense_result_row(
            "Dense embeddings",
            dense_configuration,
            dense_validation_rankings,
            validation_qrels,
            dense_runtime_seconds,
            encoding_runtime_seconds,
            embedding_dimension,
            dense_decision,
            k=k,
        ),
        _dense_result_row(
            "Sparse+dense RRF",
            hybrid_configuration,
            hybrid_validation_rankings,
            validation_qrels,
            hybrid_runtime_seconds,
            encoding_runtime_seconds,
            embedding_dimension,
            hybrid_decision,
            k=k,
        ),
    ])

    dense_submission_path = SUBMISSIONS_DIR / "submission-2-dense.csv"
    hybrid_submission_path = SUBMISSIONS_DIR / "submission-3-hybrid-rrf.csv"
    assert dense_submission_path.name != "submission-1.csv"
    assert hybrid_submission_path.name != "submission-1.csv"
    write_submission(
        dense_test_rankings,
        test_queries,
        documents,
        dense_submission_path,
    )
    write_submission(
        hybrid_test_rankings,
        test_queries,
        documents,
        hybrid_submission_path,
    )
    return results, {
        "dense_only": dense_validation_rankings,
        "sparse_dense_rrf": hybrid_validation_rankings,
    }


def _build_candidate_pool_with_metadata(
    query_rows: pd.DataFrame,
    documents: pd.DataFrame,
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    candidate_k: int = RERANK_CANDIDATE_K,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, dict[str, float]]],
]:
    sparse_rankings, sparse_scores, sparse_ranks = retrieve_tfidf_with_metadata(
        query_rows,
        documents,
        analyzer="char",
        ngram_range=(3, 5),
        title_weight=2,
        k=candidate_k,
    )
    dense_rankings, dense_scores, dense_ranks = retrieve_dense_with_metadata(
        query_embeddings,
        document_embeddings,
        query_rows["query_id"],
        documents["document_id"],
        k=candidate_k,
    )
    candidate_rankings = fuse_rankings(
        [sparse_rankings, dense_rankings],
        k=candidate_k,
    )
    retrieval_metadata = {
        "sparse_score": sparse_scores,
        "dense_score": dense_scores,
        "sparse_rank": sparse_ranks,
        "dense_rank": dense_ranks,
    }
    return candidate_rankings, retrieval_metadata


def _candidate_gate(
    candidate_rankings: dict[str, list[str]],
    qrels: pd.DataFrame,
    candidate_k: int = RERANK_CANDIDATE_K,
    candidate_model_id: str | None = None,
    candidate_model_card_url: str | None = None,
) -> dict[str, object]:
    candidate_recall_at_5 = positive_recall_at_k(
        candidate_rankings, qrels, k=5
    )
    candidate_recall_at_50 = positive_recall_at_k(
        candidate_rankings, qrels, k=50
    )
    candidate_recall_at_100 = positive_recall_at_k(
        candidate_rankings, qrels, k=100
    )
    gate_passed = all(
        recall >= RERANK_GATE_MIN_RECALL
        and recall - candidate_recall_at_5 >= RERANK_GATE_MIN_MARGIN_OVER_TOP5
        for recall in (candidate_recall_at_50, candidate_recall_at_100)
    )
    gate = {
        "profile": PROFILE,
        "candidate_k": candidate_k,
        "candidate_model_id": candidate_model_id,
        "candidate_model_card_url": candidate_model_card_url,
        "positive_recall@5": candidate_recall_at_5,
        "positive_recall@50": candidate_recall_at_50,
        "positive_recall@100": candidate_recall_at_100,
        "gate_min_recall": RERANK_GATE_MIN_RECALL,
        "gate_min_margin_over_top5": RERANK_GATE_MIN_MARGIN_OVER_TOP5,
        "gate_passed": gate_passed,
    }
    print("rerank candidate gate:", gate)
    return gate


def _rerank_decision(method: str, ndcg: float, baseline_ndcg: float) -> str:
    if ndcg > baseline_ndcg:
        return (
            f"adopted; {method} improves grouped-validation nDCG@5 "
            f"({ndcg:.6f} > sparse+dense RRF {baseline_ndcg:.6f})"
        )
    return (
        f"not adopted; {method} does not improve grouped-validation nDCG@5 "
        f"({ndcg:.6f} <= sparse+dense RRF {baseline_ndcg:.6f})"
    )


def _rerank_result_row(
    method: str,
    rankings: dict[str, list[str]],
    validation_qrels: pd.DataFrame,
    runtime_seconds: float,
    encoding_runtime_seconds: float,
    gate: dict[str, object],
    configuration: str,
    model_identifier: str,
    model_card_url: str | None,
    license_url: str,
    decision: str,
    candidate_model_id: str | None = None,
    candidate_model_card_url: str | None = None,
    candidate_k: int = RERANK_CANDIDATE_K,
) -> dict[str, object]:
    metrics = evaluate_rankings(rankings, validation_qrels, k=5)
    return {
        "method": method,
        "profile": PROFILE,
        "split": "grouped_validation_80_20",
        "seed": SEED,
        "parameters": configuration,
        "configuration": configuration,
        "mean_ndcg@5": mean_ndcg_at_k(metrics, k=5),
        "positive_recall@5": positive_recall_at_k(
            rankings, validation_qrels, k=5
        ),
        "candidate_recall@5": gate["positive_recall@5"],
        "candidate_recall@50": gate["positive_recall@50"],
        "candidate_recall@100": gate["positive_recall@100"],
        "gate_min_recall": gate["gate_min_recall"],
        "gate_min_margin_over_top5": gate["gate_min_margin_over_top5"],
        "gate_passed": gate["gate_passed"],
        "runtime_seconds": runtime_seconds,
        "encoding_runtime_seconds": encoding_runtime_seconds,
        "model_identifier": model_identifier,
        "model_card_url": model_card_url,
        "candidate_model_id": candidate_model_id,
        "candidate_model_card_url": candidate_model_card_url,
        "candidate_k": candidate_k,
        "license_url": license_url,
        "kaggle_status": "generated locally; not submitted",
        "decision": decision,
        "interpretation": (
            "Validation reranking uses the fixed grouped holdout and the "
            "same sparse-best+dense candidate pool; test labels are unused."
        ),
    }


def run_reranking_experiments(
    train_queries: pd.DataFrame,
    validation_queries: pd.DataFrame,
    test_queries: pd.DataFrame,
    documents: pd.DataFrame,
    qrels: pd.DataFrame,
    train_query_ids: set[str],
    validation_qrels: pd.DataFrame,
    candidate_k: int = RERANK_CANDIDATE_K,
) -> pd.DataFrame:
    """Run gated LambdaMART and optional cross-encoder validation."""
    if PROFILE not in RERANK_PROFILES or DENSE_MODEL_NAME is None:
        raise RuntimeError(
            "Reranking experiments require AGRO_RAG_PROFILE=rerank_local, "
            "rerank_cloud, or cross_encoder_cloud."
        )
    _invalidate_rerank_artifacts()
    _check_rerank_dependencies()
    dense_model_name = str(DENSE_MODEL_NAME)
    candidate_k = max(candidate_k, RERANK_CANDIDATE_K)
    all_queries = pd.concat([train_queries, test_queries], ignore_index=True)

    encoding_started = time.perf_counter()
    document_embeddings, all_query_embeddings = encode_documents_and_queries(
        dense_model_name,
        documents,
        all_queries,
    )
    encoding_runtime_seconds = time.perf_counter() - encoding_started
    np.save(OUTPUT_DIR / "rerank_document_embeddings.npy", document_embeddings)
    np.save(OUTPUT_DIR / "rerank_query_embeddings.npy", all_query_embeddings)

    candidate_rankings, retrieval_metadata = (
        _build_candidate_pool_with_metadata(
            all_queries,
            documents,
            document_embeddings,
            all_query_embeddings,
            candidate_k=candidate_k,
        )
    )
    gate = _candidate_gate(
        candidate_rankings,
        validation_qrels,
        candidate_k=candidate_k,
        candidate_model_id=dense_model_name,
        candidate_model_card_url=DENSE_MODEL_CARD_URL,
    )
    pd.DataFrame([gate]).to_csv(
        OUTPUT_DIR / "rerank_candidate_gate.csv", index=False
    )
    if not gate["gate_passed"]:
        print(
            "reranking skipped: candidate recall did not materially exceed "
            "top-5 under the documented gate"
        )
        return pd.DataFrame()

    validation_features, validation_labels = build_candidate_features(
        validation_queries,
        documents,
        candidate_rankings,
        retrieval_metadata,
    )
    test_features, test_labels = build_candidate_features(
        test_queries,
        documents,
        candidate_rankings,
        retrieval_metadata,
    )
    assert not validation_labels.any()
    assert not test_labels.any()

    train_features: pd.DataFrame | None = None
    train_labels: np.ndarray | None = None
    fit_train_features: pd.DataFrame | None = None
    fit_train_labels: np.ndarray | None = None
    fit_query_groups: list[int] = []
    if _lambdamart_enabled():
        train_features, train_labels = build_candidate_features(
            train_queries,
            documents,
            candidate_rankings,
            retrieval_metadata,
            qrels=qrels,
        )
        assert train_features is not None
        assert train_labels is not None
        train_mask = train_features["query_id"].isin(train_query_ids).to_numpy()
        fit_train_features = train_features.loc[train_mask].reset_index(drop=True)
        fit_train_labels = train_labels[train_mask]
        fit_query_order = [
            str(query_id)
            for query_id in train_queries["query_id"]
            if str(query_id) in train_query_ids
        ]
        assert fit_train_features is not None
        assert fit_train_labels is not None
        fit_group_sizes = fit_train_features.groupby("query_id", sort=False).size()
        fit_query_groups = [
            int(fit_group_sizes.loc[query_id]) for query_id in fit_query_order
        ]
        assert sum(fit_query_groups) == len(fit_train_features)
        assert len(fit_train_labels) == len(fit_train_features)

    validation_rankings = {
        str(query_id): candidate_rankings[str(query_id)]
        for query_id in validation_queries["query_id"]
    }
    baseline_ndcg = mean_ndcg_at_k(
        evaluate_rankings(validation_rankings, validation_qrels, k=5),
        k=5,
    )
    result_rows: list[dict[str, object]] = []
    validation_scores: dict[str, float] = {}

    if _lambdamart_enabled():
        assert fit_train_features is not None
        assert fit_train_labels is not None
        lambdamart_started = time.perf_counter()
        lambdamart_predictions = rerank_with_lambdamart(
            fit_train_features,
            fit_train_labels,
            fit_query_groups,
            validation_features,
        )
        lambdamart_runtime_seconds = time.perf_counter() - lambdamart_started
        lambdamart_rankings = _rank_candidates_by_score(
            validation_features,
            lambdamart_predictions,
            k=5,
        )
        lambdamart_ndcg = mean_ndcg_at_k(
            evaluate_rankings(lambdamart_rankings, validation_qrels, k=5),
            k=5,
        )
        validation_scores["lambdamart"] = lambdamart_ndcg
        result_rows.append(_rerank_result_row(
            "LambdaMART",
            lambdamart_rankings,
            validation_qrels,
            lambdamart_runtime_seconds,
            encoding_runtime_seconds,
            gate,
            f"candidate_model_id={dense_model_name}; candidate_k={candidate_k}; "
            "objective=lambdarank; eval_at=[5]; n_estimators=150; "
            "learning_rate=0.05; num_leaves=15; max_depth=5; "
            "min_child_samples=20; reg_lambda=1.0",
            "lightgbm.LGBMRanker(objective=lambdarank)",
            None,
            LIGHTGBM_LICENSE_URL,
            _rerank_decision("LambdaMART", lambdamart_ndcg, baseline_ndcg),
            candidate_model_id=dense_model_name,
            candidate_model_card_url=DENSE_MODEL_CARD_URL,
            candidate_k=candidate_k,
        ))

    cross_encoder_rankings: dict[str, list[str]] | None = None
    if _cross_encoder_enabled():
        cross_encoder_started = time.perf_counter()
        cross_encoder_rankings = rerank_with_cross_encoder(
            CROSS_ENCODER_MODEL_NAME,
            validation_queries,
            documents,
            validation_rankings,
            k=5,
        )
        cross_encoder_runtime_seconds = (
            time.perf_counter() - cross_encoder_started
        )
        cross_encoder_ndcg = mean_ndcg_at_k(
            evaluate_rankings(cross_encoder_rankings, validation_qrels, k=5),
            k=5,
        )
        validation_scores["cross_encoder"] = cross_encoder_ndcg
        result_rows.append(_rerank_result_row(
            "Cross-encoder",
            cross_encoder_rankings,
            validation_qrels,
            cross_encoder_runtime_seconds,
            encoding_runtime_seconds,
            gate,
            f"candidate_model_id={dense_model_name}; candidate_k={candidate_k}; "
            f"model={CROSS_ENCODER_MODEL_NAME}; k=5",
            CROSS_ENCODER_MODEL_NAME,
            CROSS_ENCODER_MODEL_CARD_URL,
            CROSS_ENCODER_LICENSE_URL,
            _rerank_decision(
                "cross-encoder", cross_encoder_ndcg, baseline_ndcg
            ),
            candidate_model_id=dense_model_name,
            candidate_model_card_url=DENSE_MODEL_CARD_URL,
            candidate_k=candidate_k,
        ))
    else:
        print("cross-encoder validation disabled for selected rerank profile")

    selected_method = max(
        validation_scores,
        key=lambda method: (validation_scores[method], method),
    )
    if validation_scores[selected_method] <= baseline_ndcg:
        selected_method = "none"
    for row in result_rows:
        row["selection"] = (
            "selected" if str(row["method"]).lower().replace("-", "_")
            == selected_method else "not selected"
        )
    print({
        "rerank_baseline_ndcg@5": baseline_ndcg,
        "selected_method": selected_method,
    })

    if _lambdamart_enabled():
        assert train_features is not None
        assert train_labels is not None
        # Model selection is complete. Final LambdaMART fits all 308 training
        # query groups only; validation and test candidates never enter fitting.
        all_group_sizes = train_features.groupby("query_id", sort=False).size()
        all_query_groups = [
            int(all_group_sizes.loc[str(query_id)])
            for query_id in train_queries["query_id"]
        ]
        assert sum(all_query_groups) == len(train_features)
        final_started = time.perf_counter()
        final_lambdamart_predictions = rerank_with_lambdamart(
            train_features,
            train_labels,
            all_query_groups,
            test_features,
        )
        final_lambdamart_runtime_seconds = time.perf_counter() - final_started
        final_lambdamart_rankings = _rank_candidates_by_score(
            test_features,
            final_lambdamart_predictions,
            k=candidate_k,
        )
        lambdamart_submission_path = (
            SUBMISSIONS_DIR / "submission-4-lambdamart.csv"
        )
        assert lambdamart_submission_path.name not in {
            "submission-1.csv",
            "submission-2-dense.csv",
            "submission-3-hybrid-rrf.csv",
        }
        write_submission(
            final_lambdamart_rankings,
            test_queries,
            documents,
            lambdamart_submission_path,
        )
        print({
            "final_lambdamart_runtime_seconds": final_lambdamart_runtime_seconds,
            "wrote": str(lambdamart_submission_path),
        })

    if cross_encoder_rankings is not None:
        final_cross_encoder_rankings = rerank_with_cross_encoder(
            CROSS_ENCODER_MODEL_NAME,
            test_queries,
            documents,
            {
                str(query_id): candidate_rankings[str(query_id)]
                for query_id in test_queries["query_id"]
            },
            k=candidate_k,
        )
        cross_encoder_submission_path = (
            SUBMISSIONS_DIR / "submission-5-cross-encoder.csv"
        )
        assert cross_encoder_submission_path.name not in {
            "submission-1.csv",
            "submission-2-dense.csv",
            "submission-3-hybrid-rrf.csv",
        }
        write_submission(
            final_cross_encoder_rankings,
            test_queries,
            documents,
            cross_encoder_submission_path,
        )
        print({"wrote": str(cross_encoder_submission_path)})

    return pd.DataFrame(result_rows)


sparse_experiment_results, sparse_rankings = run_sparse_experiments(
    validation_queries,
    tables["documents"],
    validation_qrels,
)
assert len(sparse_experiment_results) == 7
for ranking in sparse_rankings.values():
    assert all(len(ids) == 5 for ids in ranking.values())
    assert all(len(ids) == len(set(ids)) for ids in ranking.values())

sparse_experiment_results.to_csv(
    OUTPUT_DIR / "sparse_experiments.csv", index=False
)
print(sparse_experiment_results.to_string(index=False))

# %% [markdown]
# ## Opt-in cloud dense and hybrid retrieval
#
# Dense execution is intentionally skipped for the default `sparse_local`
# profile. On Kaggle/cloud, set `AGRO_RAG_PROFILE=dense_cloud`; this evaluates
# dense-only and `char_tfidf_title_2` + dense RRF on the same grouped validation
# IDs, saves embeddings below `scripts/outputs/`, and writes validated
# method-specific test submissions without touching
# `scripts/submissions/submission-1.csv`.

# %%
if PROFILE == "dense_cloud":
    dense_experiment_results, dense_rankings = run_dense_experiments(
        validation_queries,
        tables["documents"],
        validation_qrels,
        tables["test_queries"],
    )
    dense_experiment_results.to_csv(
        OUTPUT_DIR / "dense_experiments.csv", index=False
    )
    print(dense_experiment_results.to_string(index=False))
elif PROFILE in RERANK_PROFILES:
    reranking_experiment_results = run_reranking_experiments(
        tables["train_queries"],
        validation_queries,
        tables["test_queries"],
        tables["documents"],
        tables["qrels"],
        train_query_ids,
        validation_qrels,
    )
    if reranking_experiment_results.empty:
        print("reranking produced no result artifact")
    else:
        reranking_experiment_results.to_csv(
            OUTPUT_DIR / "reranker_experiments.csv", index=False
        )
        print(reranking_experiment_results.to_string(index=False))

# %% [markdown]
# ## Opt-in gated reranking
#
# `rerank_local` and `rerank_cloud` build candidates for all 308 labeled train
# queries, the fixed 56-query validation holdout, and 200 test queries from the
# same `char_tfidf_title_2` + dense RRF pool (`candidate_k=100`). Reranking runs
# only when both candidate recall@50 and candidate recall@100 are at least 0.80
# and each exceeds candidate recall@5 by at least 0.20. The
# `cross_encoder_cloud` profile is cross-encoder-only and does not require or
# execute LightGBM. `AGRO_RAG_RUN_CROSS_ENCODER=1` in a LambdaMART rerank
# profile additionally enables disclosed cross-encoder inference.

# %% [markdown]
# ## Final first-submission retrieval
#
# Character TF-IDF with duplicated titles was selected from grouped validation.
# This final lane uses local documents and test queries only; no hidden labels
# enter submission construction.

# %%
final_submission_started = time.perf_counter()
final_test_rankings = retrieve_tfidf(
    tables["test_queries"],
    tables["documents"],
    analyzer="char",
    ngram_range=(3, 5),
    title_weight=2,
    k=5,
)
final_submission_runtime_seconds = (
    time.perf_counter() - final_submission_started
)
submission = write_submission(
    final_test_rankings,
    tables["test_queries"],
    tables["documents"],
    OUTPUT_DIR / "submission.csv",
)
print(submission.head(10))
print(f"wrote {len(submission)} rows to {OUTPUT_DIR / 'submission.csv'}")
print(f"final retrieval runtime: {final_submission_runtime_seconds:.6f}s")

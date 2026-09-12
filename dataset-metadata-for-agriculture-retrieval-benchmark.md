# Agriculture Retrieval Benchmark (Sub-Saharan Africa)

A document-retrieval benchmark in the style of TREC / BEIR. Given a smallholder farmer's question, the task is to retrieve and rank the most relevant agricultural-extension documents from a fixed knowledge base. This is the retrieval half of a RAG system — the part that decides which documents an LLM should read before answering.

## TL;DR

| Item | Value |
|---|---|
| Task | Rank documents by relevance to each query |
| Corpus | 695 documents (crop diseases, pests, nutrient deficiencies, soil, climate, fertiliser) |
| Train queries | 308 (with relevance labels) |
| Test queries | 200 (labels hidden) |
| Metric (both leaderboards) | nDCG@5 |
| Submission | Long format: `QueryId, DocumentId` — top 5 docs per query; row order = rank |
| TF-IDF baseline | nDCG@5 = 0.551 |

## The Task

Each query is a natural farmer question (e.g. *"Why are my maize leaves turning yellow?"*). Several documents in the corpus are relevant, with **graded relevance**:

| Score | Meaning |
|---|---|
| **3 — Perfect** | Directly answers the question (e.g. the identification facet of nitrogen deficiency in maize, including all agro-zone variants) |
| **2 — Relevant** | A genuinely complementary facet of the same topic (e.g. symptoms can support identification). A document about the right crop and problem but the wrong intent is *not* automatically relevant |
| **1 — Marginal** | The same issue on a different crop |
| **0 — Not relevant** | Includes hard negatives that look lexically similar but answer the wrong intent (e.g. prevention instead of causes), as well as documents about a confusable problem — these separate intent-aware retrieval from keyword matching |

Your system is scored on how well its top-5 ranking matches these judgements.

## Files

### Provided to competitors (`public/`)

| File | Rows | Description |
|---|---|---|
| `documents.csv` | 695 | The knowledge base to retrieve from |
| `train_queries.csv` | 308 | Training queries with their relevant docs |
| `qrels_train.csv` | 4,194 | Graded relevance labels for the training queries |
| `test_queries.csv` | 200 | Test queries without answers — what you score against |
| `sample_submission.csv` | 1,000 | Exact submission format (200 queries × 5 rows) |
| `baseline_submission.csv` | 1,000 | A TF-IDF baseline you must beat (nDCG@5 = 0.551) |

### Column Schemas

**`documents.csv`**

| Column | Type | Description |
|---|---|---|
| `document_id` | str | Unique document id (use this value as `DocumentId` in submissions) |
| `title` | str | Document title |
| `text` | str | Document body (a short extension factsheet) |
| `source` | str | Attributed publisher (FAO, CGIAR, Plantwise, IITA, ICRISAT, AGRA, …) |
| `crop` | str | Primary crop, or `(general)` for soil/climate topics |
| `country` | str | Example country context |
| `origin` | str | `synthetic` or `llm_grounded` (see [How It Was Built](#licensing--attribution)) |
| `source_url` | str | Source link for grounded docs; blank for synthetic rows; use with `license` for reuse |
| `license` | str | Exact per-row license/reuse value; use with `source_url` to determine obligations |

### Observed License Values

The current 695-row corpus contains these exact observed `license` values:

| `license` value | Rows |
|---|---:|
| `synthetic (CC0)` | 637 |
| `CC-BY-4.0` | 55 |
| `CC-BY-SA-4.0` | 1 |
| `CC0-1.0` | 1 |
| `CC-BY-3.0` | 1 |

The per-row `license` and `source_url` fields govern reuse; do not infer terms
from `origin` or `source` alone. CC-BY rows require retained attribution, and
`CC-BY-SA-4.0` additionally requires adaptations to be shared under the same
terms where applicable.

**`train_queries.csv`**

| Column | Type | Description |
|---|---|---|
| `query_id` | int | Unique query id (train ids start at 1) |
| `query` | str | The farmer's question |
| `positive_docs` | str | Space-separated `document_id`s with relevance ≥ 1 (a convenience helper) |

**`test_queries.csv`**

| Column | Type | Description |
|---|---|---|
| `query_id` | int | Unique query id (test ids start at 1001; submit as `QueryId`) |
| `query` | str | The farmer's question |

**`qrels_train.csv`**

| Column | Type | Description |
|---|---|---|
| `query_id` | str | Query id |
| `document_id` | str | Document id |
| `relevance` | double | Graded relevance: 3, 2, 1, or 0 |

**`sample_submission.csv`** and **`baseline_submission.csv`**

| Column | Type | Description |
|---|---|---|
| `QueryId` | str | Test query id (from `test_queries.csv`) |
| `DocumentId` | str | A retrieved document id |

> Row order within each `QueryId` group is the predicted ranking: first row = rank 1 (best), fifth row = rank 5. Both files have exactly 5 rows per test query (1,000 rows total).

## Submission Format

A long-format CSV with one row per retrieved document. For each test query, output your top 5 `DocumentId`s — row order is the ranking (best document first).

```csv
QueryId,DocumentId
1001,42
1001,87
1001,19
1001,55
1001,12
1002,8
1002,64
1002,31
1002,9
1002,44
```

**Rules:**
- Submit exactly 5 rows per test `QueryId` (1,000 rows total for 200 test queries).
- First row for a query = rank 1, second = rank 2, and so on. No score column.
- Every test `QueryId` must appear.
- See `sample_submission.csv` for the exact format.
- `baseline_submission.csv` shows a working TF-IDF retriever in the same format.

## Minimal Example (Python)

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = pd.read_csv("documents.csv")
test = pd.read_csv("test_queries.csv")

vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
doc_mat = vec.fit_transform(docs["title"] + ". " + docs["text"])
sims = cosine_similarity(vec.transform(test["query"]), doc_mat)

rows = []
for i, qid in enumerate(test["query_id"]):
    for rank in sims[i].argsort()[::-1][:5]:          # best-first order
        rows.append({"QueryId": str(qid),
                     "DocumentId": str(docs.iloc[rank]["document_id"])})

pd.DataFrame(rows).to_csv("submission.csv", index=False)
```

## Evaluation

**Both leaderboards (Public and Private):** nDCG@5 (Normalised Discounted Cumulative Gain) — rewards putting higher-relevance documents nearer the top of your top-5. The same metric scores both; the only difference is which queries fall in each split.

nDCG@5 is chosen as the single metric because the dataset has graded relevance and multiple relevant docs per query — it captures both:
1. Whether you found the relevant documents, and
2. Whether you ranked the most relevant ones first.

The provided TF-IDF `baseline_submission.csv` scores **nDCG@5 = 0.551** on the hidden test labels — a bar that is clearly above random but leaves headroom for semantic and hybrid retrievers.

## Licensing & Attribution

- **Synthetic documents** (`origin = synthetic`) are original content created for this competition and released under **CC0** (public domain). The observed `synthetic (CC0)` value covers 637 rows. The `source` field on these rows is a stylistic attribution to the type of organisation that publishes such guidance — it is *not* a claim that the text was copied from them.
- **Grounded documents** (`origin = llm_grounded`) are derived from CC-BY / CC0 open-access materials (chiefly CGIAR/CGSpace). The per-row `source_url` and `license` fields govern reuse and attribution; CC-BY-SA-4.0 also carries a share-alike obligation for adaptations. License filtering excludes NoDerivatives (ND) and NonCommercial (NC) materials.
- When in doubt, consult the exact `license` and `source_url` values on each row.

## Intended Use & Limitations

- **Intended use:** Education and benchmarking of retrieval / RAG systems. Built for the TRI AI Saturdays cohort competition.
- **Not agronomic advice.** Synthetic and LLM-rewritten text may contain simplifications or errors and must not be used for real farming decisions.
- **Geographic focus:** Sub-Saharan African smallholder agriculture; not representative of other regions or farming systems.
- **Reproducibility:** The dataset is generated internally, which caches all source fetches and LLM rewrites so runs are deterministic.

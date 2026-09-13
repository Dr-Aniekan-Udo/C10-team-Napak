# Retrieval Methods Research Ledger

| Method | Source | Adoption | Model/license | Validation result | Decision |
|---|---|---|---|---:|---|
| TF-IDF/BM25 | BEIR, arXiv:2104.08663 | Robust lexical reference | No pretrained model | grouped validation; best nDCG@5 0.543535; best positive recall@5 0.379974 | local sparse candidate: character TF-IDF, title weight 2 |
| Dense embeddings | DPR, arXiv:2004.04906; SBERT, arXiv:1908.10084 | Local CPU semantic lane; adopted | `sentence-transformers/all-MiniLM-L6-v2`; Apache-2.0 ([license](https://www.apache.org/licenses/LICENSE-2.0)); [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | local CPU grouped validation; dimension=384; runtime=519.993614s; encoding=519.752736s; nDCG@5=0.6583537503; positive/candidate recall@5=0.3948979592; candidate recall@50=0.8531462585; @100=0.9700042517 | adopted from same grouped-validation evidence; improves sparse baseline nDCG@5 0.543535 |
| Sparse+dense RRF | Light Hybrid Retrievers, arXiv:2210.01371 | Local CPU candidate fusion; adopted | `char_tfidf_title_2` + `sentence-transformers/all-MiniLM-L6-v2`; Apache-2.0 ([license](https://www.apache.org/licenses/LICENSE-2.0)); [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | local CPU grouped validation; dimension=384; runtime=524.131610s; encoding=519.752736s; nDCG@5=0.6438034025; positive/candidate recall@5=0.3608843537; candidate recall@50=0.9160289116; @100=0.9854272959 | adopted from same grouped-validation evidence; improves sparse baseline nDCG@5 0.543535 |
| LambdaMART reranker | LightGBM `LGBMRanker` | Gated learned reranker; validated locally; retained comparison | LightGBM 4.7.0; `eval_at=[5]` passed to `fit`; [license](https://github.com/microsoft/LightGBM/blob/master/LICENSE); candidate model `sentence-transformers/all-MiniLM-L6-v2`; candidate_k=100 | local `rerank_local` grouped validation; gate passed; nDCG@5=0.7584923684; positive recall@5=0.4682823129; validation rerank runtime=6.2054824s; candidate encoding runtime=99.1849300s | cross-encoder selected over LambdaMART on same holdout |
| Cross-encoder | BEIR, arXiv:2104.08663 | Candidate reranker; selected on local grouped validation | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M parameters; ~86.7 MiB weights; max 512 pair tokens); Apache-2.0 ([license](https://www.apache.org/licenses/LICENSE-2.0)); [model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | local CPU `cross_encoder_cloud` grouped validation; gate passed; nDCG@5=0.8484345554; positive recall@5=0.4748299320; validation rerank runtime=573.8966775s; candidate encoding runtime=94.7210773s | selected over LambdaMART nDCG@5 0.7584923684; `submission-5-cross-encoder.csv` recorded as highest-scoring submitted artifact; README leaderboard values are external/project-record snapshots, not locally recomputed hidden-label evidence |
| ColBERT | arXiv:2004.12832 | Deferred late interaction | pretrained model not selected | not run | deferred |
| SPLADE | arXiv:2107.05720 | Deferred learned sparse expansion | pretrained model not selected | not run | deferred |

## Experiment Notes

Local profile ran on supplied `data/` only with Python/NumPy/pandas/scikit-learn,
CPU-only, seed `42`, and no pretrained model. Validation uses the deterministic
grouped 80/20 split: 252 training queries and 56 validation queries from 122
positive-document families. Positive recall@5 is macro-average recall over each
validation query's positive qrels (`relevance > 0`). Runtime is wall-clock
retrieval or fusion time in seconds on the local machine.

Task 4 dense execution is opt-in with `AGRO_RAG_PROFILE=dense_cloud`. It
completed on local CPU; GPU/cloud comparison remains unrun. Default
`sparse_local` does not import either dense package. The disclosed default is
`sentence-transformers/all-MiniLM-L6-v2` (384-dimensional normalized embeddings,
Apache-2.0, [license](https://www.apache.org/licenses/LICENSE-2.0), [model
card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). On a
successful dense run, the profile writes
`scripts/outputs/dense_experiments.csv`,
`scripts/outputs/dense_document_embeddings.npy`, and
`scripts/outputs/dense_query_embeddings.npy`; embeddings never write outside
`scripts/outputs/`. The completed local run validated and wrote two test artifacts:
`scripts/submissions/submission-2-dense.csv` and
`scripts/submissions/submission-3-hybrid-rrf.csv`; both were generated locally and not
submitted. `scripts/submissions/submission-1.csv` is never overwritten. Dense and hybrid
adoption decisions compare only against
`char_tfidf_title_2` on the same grouped validation IDs; underperformance keeps
the sparse selection.

## Task 5 Gated Reranking

Reranking is opt-in through `AGRO_RAG_PROFILE=rerank_local` or
`rerank_cloud`; both profiles run LambdaMART. `cross_encoder_cloud` is a
cross-encoder-only profile and does not require LightGBM or execute LambdaMART.
`AGRO_RAG_RUN_CROSS_ENCODER=1` within a rerank profile additionally enables
`cross-encoder/ms-marco-MiniLM-L-6-v2` inference. The candidate pool is built
once for all 308 labeled training queries, the fixed 56-query validation
holdout, and 200 test queries from `char_tfidf_title_2` + dense RRF at the
actual `candidate_k=100`. Candidate model ID is
`sentence-transformers/all-MiniLM-L6-v2`; no model revision or hash is claimed.
Application configuration keeps this candidate model separate from
`AGRO_RAG_RANK_MODEL` through `AGRO_RAG_CANDIDATE_MODEL`, whose documented
default is `sentence-transformers/all-MiniLM-L6-v2`. This value records dense
candidate provenance/configuration for an injected dense or sparse+dense
`CandidateGenerator`; it is not silently loaded or used by the default factory.
The app-safe default candidate generator intentionally uses validated sparse
char-TF-IDF title-weight-2 behavior, so local/offline ranker construction has
no candidate-model download. Dense or sparse+dense generators must be injected
explicitly and own their model loading.
Candidate recall is printed and recorded before reranking.

The explicit gate requires both candidate recall@50 and candidate recall@100 to
be at least `0.80` and at least `0.20` above candidate recall@5. The completed
gate passed with recall@5=`0.3608843537`, recall@50=`0.9160289116`, and
recall@100=`0.9854272959`. Validation candidates are never fitted; final
LambdaMART fitting, after selection, uses all 308 training query groups.

Completed command:

```text
AGRO_RAG_PROFILE=rerank_local AGRO_RAG_DATA_DIR="$PWD/data" uv run python scripts/notebooks/01_retrieval_experiments.py
```

```text
AGRO_RAG_PROFILE=cross_encoder_cloud AGRO_RAG_DATA_DIR="$PWD/data" uv run python scripts/notebooks/01_retrieval_experiments.py
```

The final local `rerank_local` run used existing LightGBM 4.7.0 from the
`rerank` extra plus already-installed sentence-transformers/Torch. LambdaMART
validation returned nDCG@5=`0.7584923684` and positive recall@5=`0.4682823129`,
with validation rerank runtime `6.2054824s` and candidate encoding runtime
`99.1849300s`. It was selected/adopted over sparse+dense RRF nDCG@5
`0.6438034025`. The local `cross_encoder_cloud` run then returned
nDCG@5=`0.8484345554` and positive recall@5=`0.4748299320`, with cross-encoder
validation runtime `573.8966775s` and candidate encoding runtime
`94.7210773s`. It selected the cross-encoder over LambdaMART and generated
current `scripts/outputs/reranker_experiments.csv`,
`scripts/outputs/rerank_candidate_gate.csv`, and
`scripts/submissions/submission-5-cross-encoder.csv`. Submission-5 is recorded
as Experiment 1's highest-scoring submitted artifact. README records public
nDCG@5=`0.86625` and private nDCG@5=`0.83607` as external/project-record
leaderboard snapshots; they were not recomputed locally and are not hidden-label
evidence produced by this repository. Both reranker submissions 4 and 5
pass the 1,000-row contract; profile-aware invalidation preserves the other
method's submission.

| Run | Profile | Method | Split | Candidate gate | Mean nDCG@5 | Positive recall@5 | Candidate recall@50 | Candidate recall@100 | Runtime (s) | Model/license | Kaggle status | Decision |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| lambdamart_validation | rerank_local | LambdaMART | grouped_validation_80_20 | passed; @5=0.3608843537; @50=0.9160289116; @100=0.9854272959; thresholds min 0.80 and margin 0.20 | 0.7584923684 | 0.4682823129 | 0.9160289116 | 0.9854272959 | rerank=6.2054824; encoding=99.1849300 | LightGBM 4.7.0 `LGBMRanker`; `eval_at=[5]` via fit; candidate=`sentence-transformers/all-MiniLM-L6-v2`; candidate_k=100; https://github.com/microsoft/LightGBM/blob/master/LICENSE | generated locally; not submitted | retained comparison; cross-encoder selected |
| cross_encoder_validation | cross_encoder_cloud | Cross-encoder | grouped_validation_80_20 | passed; @5=0.3608843537; @50=0.9160289116; @100=0.9854272959; thresholds min 0.80 and margin 0.20 | 0.8484345554 | 0.4748299320 | 0.9160289116 | 0.9854272959 | rerank=573.8966775; encoding=94.7210773 | `cross-encoder/ms-marco-MiniLM-L-6-v2`; candidate=`sentence-transformers/all-MiniLM-L6-v2`; candidate_k=100; Apache-2.0; https://www.apache.org/licenses/LICENSE-2.0; https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 | submission-5 recorded as highest-scoring submitted artifact; README public/private values 0.86625/0.83607 are external/project-record snapshots, not locally recomputed hidden-label evidence | selected over LambdaMART 0.7584923684 |

### Limitations and interpretation

The approved exact positive-document-set grouping does not guarantee
document-disjoint splits: 34 positive document IDs overlap between the current
training and validation partitions. The single grouped holdout is therefore
selection-biased and is not independent generalization evidence. Qrels include
explicit zero rows but do not enumerate all corpus pairs; absent candidate qrels
are treated as zero under the closed-world evaluator semantics. These are
disclosed interpretation limitations, not unverified implementation failures.

| Run | Profile | Method | Split | Seed | Parameters / vocabulary | Mean nDCG@5 | Positive recall@5 | Candidate recall@5 | Runtime (s) | Model identifier | License | Kaggle status | Decision | Interpretation |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| word_tfidf_title_1 | sparse_local | TF-IDF word | grouped_validation_80_20 | 42 | analyzer=word; ngram_range=(1, 2); title_weight=1; vocabulary=13,215 | 0.526531 | 0.371312 | 0.371312 | 0.418399 | none | not applicable; no pretrained model | not submitted; validation only | not selected for first submission; lower nDCG@5 (0.526531) than selected character TF-IDF (0.543535) | Word lexical reference. |
| word_tfidf_title_2 | sparse_local | TF-IDF word | grouped_validation_80_20 | 42 | analyzer=word; ngram_range=(1, 2); title_weight=2; vocabulary=13,338 | 0.539542 | 0.379974 | 0.379974 | 0.285275 | none | not applicable; no pretrained model | not submitted; validation only | not selected for first submission; lower nDCG@5 (0.539542) than selected character TF-IDF (0.543535) | Title duplication improved word-lane validation nDCG over weight 1. |
| char_tfidf_title_1 | sparse_local | TF-IDF character | grouped_validation_80_20 | 42 | analyzer=char; ngram_range=(3, 5); title_weight=1; vocabulary=47,783 | 0.532335 | 0.359237 | 0.359237 | 1.903719 | none | not applicable; no pretrained model | not submitted; validation only | not selected for first submission; lower nDCG@5 (0.532335) than selected character TF-IDF (0.543535) | Character features improved nDCG over word weight 1 but lowered positive recall. |
| char_tfidf_title_2 | sparse_local | TF-IDF character | grouped_validation_80_20 | 42 | analyzer=char; ngram_range=(3, 5); title_weight=2; vocabulary=48,026 | 0.543535 | 0.359928 | 0.359928 | 2.038595 | none | not applicable; no pretrained model | not submitted; validation only | selected for first submission | Best mean nDCG@5; retained as local sparse candidate. |
| bm25_title_1 | sparse_local | BM25 | grouped_validation_80_20 | 42 | k1=1.5; b=0.75; title_weight=1; vocabulary=2,826 | 0.432336 | 0.310704 | 0.310704 | 1.714639 | none | not applicable; no pretrained model | not submitted; validation only | not selected for first submission; lower nDCG@5 (0.432336) than selected character TF-IDF (0.543535) | Dependency-free BM25 underperformed TF-IDF on this split. |
| bm25_title_2 | sparse_local | BM25 | grouped_validation_80_20 | 42 | k1=1.5; b=0.75; title_weight=2; vocabulary=2,826 | 0.452283 | 0.330219 | 0.330219 | 1.767732 | none | not applicable; no pretrained model | not submitted; validation only | not selected for first submission; lower nDCG@5 (0.452283) than selected character TF-IDF (0.543535) | Title duplication helped BM25 but remained below TF-IDF. |
| sparse_rrf | sparse_local | Sparse RRF | grouped_validation_80_20 | 42 | rrf_k=60; k=5; components=word_tfidf_title_1+char_tfidf_title_1+bm25_title_1 | 0.499061 | 0.373969 | 0.373969 | 0.001472 | none | not applicable; no pretrained model | not submitted; validation only | not selected for first submission; lower nDCG@5 (0.499061) than selected character TF-IDF (0.543535) | Fusion remained below the best single lane on nDCG and positive recall. |

## Task 4 Dense/Hybrid Runs

| Run | Profile | Method | Split | Seed | Model identifier | License URL | Embedding dimension | Runtime (s) | Encoding runtime (s) | nDCG@5 | Positive recall@5 | Candidate recall@5 | Candidate recall@50 | Candidate recall@100 | Decision | Kaggle status |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dense_only | dense_cloud | Dense embeddings | grouped_validation_80_20 | 42 | `sentence-transformers/all-MiniLM-L6-v2` ([model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) | Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 | 384 | 519.993614 | 519.752736 | 0.6583537503 | 0.3948979592 | 0.3948979592 | 0.8531462585 | 0.9700042517 | adopted | generated locally; not submitted |
| sparse_dense_rrf | dense_cloud | Sparse+dense RRF | grouped_validation_80_20 | 42 | `sentence-transformers/all-MiniLM-L6-v2` ([model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) + `char_tfidf_title_2` | Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 | 384 | 524.131610 | 519.752736 | 0.6438034025 | 0.3608843537 | 0.3608843537 | 0.9160289116 | 0.9854272959 | adopted | generated locally; not submitted |

Each row changes one controlled factor relative to its comparison lane: title
weight changes from 1 to 2 within each representation; character n-grams are a
separate representation lane; and sparse RRF fuses the three weight-1 lanes.
The reported decision is local validation evidence only, not a hidden-test or
Kaggle claim.

## Task 6 First Submission

Status: first submission generated locally at `scripts/outputs/submission.csv`; it was
not submitted to Kaggle, so no Kaggle score is available or claimed.

Submission method: `sparse_local` character TF-IDF fit on local `documents.csv`
and applied to all 200 local test queries. Configuration is
`analyzer=char; ngram_range=(3, 5); title_weight=2; k=5`, with sublinear TF,
cosine similarity, and deterministic document-ID tie breaking. Title text is
duplicated twice before concatenation with document body. No dense model,
reranker, external corpus, or hidden test labels were used. The selection is
based on the grouped-validation result above (`nDCG@5=0.543535`); test-label
metrics are not computed.

| Run | Profile | Split | Seed | Parameters | Runtime (s) | nDCG@5 | Candidate recall | Model identifier | License | Kaggle status | Decision |
|---|---|---|---:|---|---:|---|---|---|---|---|---|
| final_submission_char_tfidf_title_2 | sparse_local | final local test retrieval | 42 | analyzer=char; ngram_range=(3, 5); title_weight=2; k=5 | 2.026640 | not computed; test labels unavailable | not computed; test labels unavailable | none | not applicable; no pretrained model | generated locally; not submitted | use as first submission |

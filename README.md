# Tri-AI Cohort 10, Team Napak: Agricultural Extension Retrieval/RAG Benchmark

Team Napak benchmarks retrieval/RAG: ranking agricultural-extension documents before generation. Education/benchmarking; not agronomic advice.

## Dataset

Tri-AI/Kaggle: 695 documents; 308 labeled train queries; 200 unlabeled test queries; graded `qrels_train.csv`. Submit five unique in-corpus document IDs per test query (1,000 rows), in rank order. Agricultural extension: Sub-Saharan Africa.

Only supplied corpus used. Row `source_url`/`license` govern attribution/reuse. Synthetic rows are competition content; LLM-grounded rows derive from CC-BY/CC0. See [schema/licensing notes](dataset-metadata-for-agriculture-retrieval-benchmark.md). Future: larger permitted/licensed agro-education corpus from government and other authoritative agriculture sites; not completed.

## Training Pipeline

Source of truth: `scripts/notebooks/01_retrieval_experiments.py`; paired notebook derives one-way via Jupytext. [Ledger](docs/research/retrieval-methods.md) records evidence. Pipeline: normalize IDs, combine title/body text, seed `42`, grouped 80/20 split, fixed `candidate_k=100` pool. No external documents, hidden labels, or paid/API generation.

- Word/character TF-IDF use title weights 1/2; BM25 uses `k1=1.5`, `b=0.75`, title weights 1/2; sparse RRF uses `rrf_k=60`.
- Dense retrieval uses normalized 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- `lightgbm.LGBMRanker` / LambdaMART uses candidate/rank/lexical/dense/domain features.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` scores candidate pairs as an inference-only reranker.

Only LambdaMART trains; TF-IDF/BM25 fit indexes; dense/cross-encoder are pretrained inference models; RRF is rank fusion.

## Evaluation

Metric: deterministic grouped 80/20 validation nDCG@5, seed `42`. Grade-aware nDCG@5 uses `3/2/1`; positive recall counts relevance >0. Candidate gate: recall@50/@100 ≥ `0.80`, each ≥ `0.20` above recall@5. Ties use document ID. Submission checks: five unique in-corpus IDs/query, 1,000 rows; validation stays separate from test.

| Method | Model | nDCG@5 | Decision |
|---|---|---:|---|
| Word TF-IDF, title 1 | none | 0.526531 | Not selected |
| Word TF-IDF, title 2 | none | 0.539542 | Not selected |
| Character TF-IDF, title 1 | none | 0.532335 | Not selected |
| Character TF-IDF, title 2 | none | 0.543535 | Sparse baseline; submission 1 |
| BM25, title 1 | none | 0.432336 | Not selected |
| BM25, title 2 | none | 0.452283 | Not selected |
| Sparse RRF | rank fusion | 0.499061 | Not selected |
| Dense embeddings | `all-MiniLM-L6-v2` | 0.6583537503 | Retained; submission 2 submitted on Kaggle before deadline |
| Sparse+dense RRF | MiniLM + sparse | 0.6438034025 | Comparison; submission 3 generated |
| LambdaMART reranker | `LGBMRanker` / LightGBM 4.7.0 | 0.7584923684 | Comparison; submission 4 generated |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.8484345554 | Best local model at close; submission 5 not submitted due to a team technical issue |

Character TF-IDF is sparse lane; dense improved semantic coverage; RRF is comparison; LambdaMART improved ordering; cross-encoder won same holdout. The split overlaps 34 positive IDs, so it is not independent evidence. Test labels hidden; no local test score is claimed.

Records list Kaggle public `0.86625` and private `0.83607` nDCG@5 as external/project-record snapshots from deadline submissions—not local validation, hidden-label recomputation, or cross-encoder results. Submission 2 was submitted on Kaggle before the deadline; submission 5 was not submitted because of the team technical issue.

## Reproduction

From root, generate the cross-encoder test artifact first:

```powershell
uv sync --extra cross_encoder
$env:AGRO_RAG_PROFILE = "cross_encoder_cloud"
$env:AGRO_RAG_DATA_DIR = "$PWD\data"
uv run python scripts/notebooks/01_retrieval_experiments.py
```

Reads `data/test_queries.csv`, scores 200 test queries, validates five unique in-corpus IDs/query, and writes `scripts/submissions/submission-5-cross-encoder.csv` (1,000 rows); evidence in `scripts/outputs/`. Weights may download; CPU runtime may be substantial. This command does not evaluate hidden labels.

Optional profiles, in order:

```powershell
$env:AGRO_RAG_PROFILE = "sparse_local"
uv run python scripts/notebooks/01_retrieval_experiments.py

uv sync --extra dense --extra rerank
$env:AGRO_RAG_PROFILE = "dense_cloud"
uv run python scripts/notebooks/01_retrieval_experiments.py

$env:AGRO_RAG_PROFILE = "rerank_local"
uv run python scripts/notebooks/01_retrieval_experiments.py
```

Artifacts: `scripts/submissions/submission-1.csv`, `scripts/submissions/submission-2-dense.csv`, `scripts/submissions/submission-3-hybrid-rrf.csv`, `scripts/submissions/submission-4-lambdamart.csv`, `scripts/submissions/submission-5-cross-encoder.csv`. No hidden-label eval. Notebook:

```powershell
uv run jupytext --sync scripts/notebooks/01_retrieval_experiments.py
```

Never sync from `.ipynb` to `.py`.

### Secondary RAG app

```powershell
Copy-Item .env.example .env
# Edit .env: AGRO_RAG_API_BASE_URL, AGRO_RAG_API_KEY, AGRO_RAG_MODEL, AGRO_RAG_DATA_DIR
uv sync --extra app --extra cross_encoder
uv run python scripts/app.py
```

Keep `.env` local and uncommitted. `AGRO_RAG_API_KEY=null` is valid only for a keyless local OpenAI-compatible endpoint. App setup is separate from submission inference.

## Appendix

**Team: Team Napak**
> **Team Lead:**
- [Aniekan Etim Udo](https://github.com/Dr-Aniekan-Udo)
> **Team Members:**
- Okolo Collins Lfesinachi
- Oluwatosin Oluwatimilehin Olajide
- [Aina Temiloluwa](https://github.com/Temmy-bit)
> **Mentors:**
- Oluwaseun Ajayi
- Samuel Taiwo
- Adnan Adetunji

## References

- [BEIR](https://arxiv.org/abs/2104.08663)
- [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [Light Hybrid Retrievers](https://arxiv.org/abs/2210.01371)
- Burges, [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- [CrossEncoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [SPLADE](https://arxiv.org/abs/2107.05720)

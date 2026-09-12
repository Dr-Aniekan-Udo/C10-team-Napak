# Experiment Outputs

This directory stores local evidence and runtime artifacts. Curated Experiment 1
summaries are intended for initial public staging:

- `sparse_experiments.csv`
- `dense_experiments.csv`
- `reranker_experiments.csv`
- `rerank_candidate_gate.csv`

Embeddings, large candidate/test-ranking CSVs, per-query dumps, and Experiment 2
runtime artifacts remain local or ignored by policy. Existing bytes are retained
for reproduction and audit; no hidden-label test metric is inferred from them.

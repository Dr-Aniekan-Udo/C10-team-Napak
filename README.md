# C10 Team Napak: Agricultural Extension RAG

Public research repository for C10 Team Napak's agricultural-extension retrieval
work. The project addresses the retrieval half of a retrieval-augmented
generation (RAG) system: given a smallholder farmer's question, rank the most
relevant extension documents before an answer is generated.

## Problem

The supplied benchmark contains 695 agricultural-extension documents, 308
training queries with graded relevance labels, and 200 test queries whose labels
are hidden by the competition. Documents cover crop disease, pests, nutrients,
soil, fertilizer, and climate-related guidance for Sub-Saharan African
smallholder agriculture. Submissions use `QueryId,DocumentId` rows with five
unique documents per test query.

This repository is for education and retrieval/RAG benchmarking. It is not
agronomic advice. Synthetic and LLM-grounded text can contain simplifications or
errors and must not guide real farming decisions.

## Data Boundary

`data/` contains the supplied competition files and repository-authored metadata
describing them. That metadata and documentation elsewhere in the repository are
not additional corpus content. `data/` is not an unrestricted web corpus or a
collection of private stakeholder data.

- Rows with `origin=synthetic` are original competition content released under
  CC0. Their `source` values are stylistic publisher labels, not claims of text
  copied from those organizations.
- Rows with `origin=llm_grounded` derive from CC-BY/CC0 open-access material.
  Reuse must retain attribution from each row's `source_url` and `license`
  fields.
- The complete dataset boundary and column schemas are documented in
  `dataset-metadata-for-agriculture-retrieval-benchmark.md`.

## Experiment 1

`scripts/notebook/01_retrieval_experiments.py` is the source of truth for the
first submitted experiment. Its paired notebook is generated one-way with
Jupytext.

1. Load supplied files from `data/` or an explicit `AGRO_RAG_DATA_DIR`.
2. Use deterministic grouped 80/20 validation with seed `42`.
3. Compare sparse lexical retrieval, dense embeddings, sparse+dense RRF, gated
   LambdaMART, and a cross-encoder reranker.
4. Build a 100-document candidate gate before reranking and validate submission
   shape and document membership.
5. Keep validation evidence separate from unlabeled test rankings and Kaggle
   leaderboard evidence.

Selected local evidence for the cross-encoder submission:

| Evidence layer | Result | Interpretation |
|---|---:|---|
| Local grouped validation nDCG@5 | `0.8484345554` | Cross-encoder reranking on the fixed local holdout |
| Candidate recall@100 | `0.9854273` | Candidate-pool coverage diagnostic |
| Kaggle public nDCG@5 | `0.86625` | External/project-record leaderboard snapshot |
| Kaggle private nDCG@5 | `0.83607` | External/project-record leaderboard snapshot |

The Kaggle values are external/project-record post-submission leaderboard
snapshots for `scripts/submissions/submission-5-cross-encoder.csv`. They are not
local validation metrics, locally recomputed hidden-label evidence, or training
targets. Project records identify submission-5 as Experiment 1's highest-scoring
submitted artifact.

## Reproduce

Use Python 3.12 through `uv`:

```text
uv sync
AGRO_RAG_DATA_DIR=data AGRO_RAG_PROFILE=sparse_local uv run python scripts/notebook/01_retrieval_experiments.py
```

Dense and reranking profiles are opt-in and may download pretrained models or
take substantial CPU/GPU time. See `docs/research/retrieval-methods.md` for
model, license, and profile details.

For paired notebook maintenance, edit the `.py` file and run:

```text
uv run jupytext --sync scripts/notebook/01_retrieval_experiments.py
```

Never sync from `.ipynb` to `.py`.

## Repository Layout

```text
README.md
docs/
  research/
scripts/
  src/
  notebook/
  outputs/
  submissions/
data/
```

`scripts/notebook/` contains editable experiment sources and paired notebooks.
`scripts/src/` contains pure shared retrieval/evaluation helpers.
`scripts/outputs/` contains local evidence and runtime artifacts.
`scripts/submissions/` contains upload-format rankings.
`data/` contains supplied benchmark files and repository-authored metadata.

Experiment 2 is intentionally omitted from the initial public package until two
additional tests pass. This README does not claim Experiment 2 is complete.

## Artifact Policy

Initial public packaging tracks the supplied `data/`, Experiment 1 notebook 01,
the required shared source, curated Experiment 1 summaries, submissions 1--5,
documentation placeholders, and reproducibility metadata (`pyproject.toml`,
`uv.lock`, `.python-version`, and `run-jup.sh.example`).

Raw `.npy` embeddings, large candidate/test-ranking CSVs, per-query dumps,
checkpoints, model weights, caches, secrets, and local OpenCode/Jupyter runtime
files remain local or ignored. Experiment 2 is intentionally excluded from the
initial public package until two additional tests pass.

Dataset licensing follows the per-row `license` and `source_url` fields. The
pretrained retrieval models used by Experiment 1 are disclosed with their model
cards and Apache-2.0 license links in the research ledger.

## Documentation

- `docs/README.md` lists supplied and pending public documents.
- `docs/research/retrieval-methods.md` records method provenance and evidence.
- `scripts/` directory READMEs explain source, notebook, output, and submission
  boundaries.

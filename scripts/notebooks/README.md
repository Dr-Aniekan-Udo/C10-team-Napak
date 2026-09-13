# Experiment Notebooks

`.py` files are notebook source of truth. Paired `.ipynb` files are generated
for JupyterLab or Kaggle use.

- `01_retrieval_experiments.py` contains Experiment 1 retrieval and submission
  workflow.

`../app.py` is the separate Gradio Field Notes interface. It composes the
shared streaming pipeline but is not notebook source. Keep retrieval,
generation, and persistence behavior in `scripts/utilities/`; use the app's
factory injection point for offline UI tests.

Experiment 2 is intentionally omitted from the initial public package until two
additional tests pass. This README does not claim Experiment 2 is complete.

Edit Python sources, then run `uv run jupytext --sync <file>.py`. Never sync
from `.ipynb` to `.py`.

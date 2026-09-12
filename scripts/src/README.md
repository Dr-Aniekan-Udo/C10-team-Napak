# Shared Source

`_retrieval_common.py` contains small, deterministic, explicit-input helpers
shared by retrieval experiments. It loads supplied tables, computes grouped
splits and ranking metrics, validates rankings, and writes exclusive submission
or manifest files.

The module does not load project data or create output directories on import.
Keep experiment orchestration in `scripts/notebook/`.

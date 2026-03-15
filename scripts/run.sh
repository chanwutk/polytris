#!/bin/bash
# Two-pass pipeline: valid pass → evaluation + Pareto extraction → test pass → final evaluation

# Phase 1: Valid pass (full parameter grid, valid videoset only)
python scripts/p020_exec_classify.py --valid
python scripts/p022_exec_prune_polyominoes.py --valid
python scripts/p030_exec_compress.py --valid
python scripts/p040_exec_detect.py --valid
python scripts/p050_exec_uncompress.py --valid
python scripts/p060_exec_track.py --valid

# Phase 2+3: Evaluation on valid data and Pareto parameter extraction
python evaluation/p100_evaluation.py

# Phase 4: Test pass (auto-filtered to Pareto-optimal parameter combos)
python scripts/p020_exec_classify.py --test
python scripts/p022_exec_prune_polyominoes.py --test
python scripts/p030_exec_compress.py --test
python scripts/p040_exec_detect.py --test
python scripts/p050_exec_uncompress.py --test
python scripts/p060_exec_track.py --test

# Phase 5: Final evaluation (valid + test data for Pareto combos)
python evaluation/p100_evaluation.py

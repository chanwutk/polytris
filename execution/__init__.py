"""Pipeline-parallel execution runner for the polyis tracking pipeline.

A single command (``execution.main``) runs the operators p020 -> p022 -> p030
-> p040 -> p050 -> p060 concurrently for one parameter combination, scaling
CPU-bound stages (Prune, Compress) horizontally via ``mp.Process`` pools and
pinning GPU-bound stages (Classify, Detect) to specific GPUs.

The final tracking output is written to the same cache path as
``scripts/p060_exec_track.py`` so downstream evaluation scripts work
unchanged.  See ``execution/main.py`` for the CLI surface.
"""

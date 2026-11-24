# LEAP tools

Byte-identical copies of the files the experiments use from [chanwutk/tetris-leap](https://github.com/chanwutk/tetris-leap) at `70afe9d`, with that repo's `experiments/` prefix dropped:

- `evaluate.py`: LEAP's original query evaluator, run by the paper's `scripts/score_leap_queries.py --evaluator`.
- `eval/export_tracks.py` and `eval/export-leap-outputs.py`: the track exporter wrapper and the exporter the scored LEAP runs used.
- `run/run_leap.py`, `run/matched_detector.py`, `prep/motion_patterns.py`: runners for LEAP's image, which mounts LEAP's own code at `/leap`.

Completed runs and the paper's evidence record these files' sha256, so keep them unchanged.

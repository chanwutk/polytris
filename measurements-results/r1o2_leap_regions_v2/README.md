# LEAP2 region comparison

- The original detector-matched run is preserved in `../r1o2_leap_matched/`.
- LEAP2 uses the frozen revised `stop_area` definitions in `configs/` and `regions/`. Every other configuration field is identical.
- The original detector checkpoints, pseudo-labels, re-identification models, motion libraries, clips, and inference implementation are reused. Preparation is mounted read-only.
- Each dataset runs on one GPU with its original CPU subset; the three datasets were executed concurrently on bombe GPUs 0, 2, and 5.
- `inputs.json` records input hashes; `raw/` contains the new outputs and clean-launch records; `summary/` holds native speed files; `comparison/results.json` holds the original/LEAP2 comparison.
- HOTA uses unchanged TrackEval and the original root ground truth. Raw, linear, and assigned-pattern readings are preserved; the main comparison chooses the better of linear and pattern for each version.
- Native selection, count, and top-10 use the original LEAP evaluator class. All three original query scores were reproduced before comparison. Count is MAPE; FPS is native inference time minus decode time.
- The original outputs were not overwritten. `../leap_canonical.json` selects this reviewed-region version as canonical LEAP for Amsterdam, CalDoT1, and CalDoT2. B3D results retain their existing selection.
- `canonical-archive/` preserves the original matched-run summaries and paper evidence before promotion. The paper's primary LEAP points, query selections, Pareto plots, and amortization use the reviewed version; the original/reviewed comparison remains available.

The refresh can be reproduced in ace's `polyis` container using the paper scripts staged in `paper-workspace/`:

```sh
python measurements-results/r1o2_leap_regions_v2/paper-workspace/scripts/refresh_leap_matched.py --tetris-root /polyis --matched-summary /polyis/measurements-results/r1o2_leap_regions_v2/canonical-archive/matched_summary
python measurements-results/r1o2_leap_regions_v2/paper-workspace/scripts/render_revised_hota.py
python measurements-results/r1o2_leap_regions_v2/paper-workspace/scripts/generate_revision_figures.py
python measurements-results/r1o2_leap_regions_v2/paper-workspace/scripts/generate_query_pareto.py
```

`verify_canonical.py` checks the selected track paths, source hashes, unchanged original CSVs, unchanged preparation, and the three-record query-data replacement. The reviewed native-interval summary is saved in `summary/native_intervals.json`.

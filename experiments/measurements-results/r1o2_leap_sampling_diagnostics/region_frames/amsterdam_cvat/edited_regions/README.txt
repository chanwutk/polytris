Amsterdam edited LEAP regions, revision 2026-10-31 23:25:48 UTC

cvat_export.zip and annotations.xml retain the latest supplied CVAT export.
stop_area.yaml contains all five edited boxes, with exact exported coordinates in native 1280 x 720 pixels.
Display region numbering follows CVAT XML order.
The entire previous revision is preserved under ../region_history/2026-10-31_22-36-58/.
regions.png shows boxes on the original frame; regions_with_trajectories.png adds ground-truth trajectories.
pattern_review/patterns_with_regions.png shows learned LEAP paths and uncovered endpoints.

Read-only motion-library check: 25/33 eligible previously; 29/33 now. No previously eligible patterns were lost.
Remaining ineligible pattern IDs: 2, 3, 13, 21 (zero-based). These include mid-road endpoints.
pattern_check.json records every endpoint, original/previous/new region membership, and source-library hash.
User retained the current CalDoT1 and CalDoT2 region geometries because their uncovered endpoints are mid-road, not exits.
This is a region configuration fragment; live LEAP extraction configurations and paper numbers were not changed.
No extraction, training, runtime measurement, or accuracy evaluation was rerun.

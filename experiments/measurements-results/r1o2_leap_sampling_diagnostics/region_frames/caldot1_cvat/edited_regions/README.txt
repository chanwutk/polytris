CalDoT1 edited LEAP regions

Imported from CVAT job 1. cvat_export.zip and annotations.xml preserve the user export.
stop_area.yaml contains all 2 region boxes, with exact exported coordinates in native 720 x 480 pixels.
Box numbering follows CVAT XML order; edited/recreated boxes may be reordered relative to the original preview.
regions.png shows the edited regions on the original frame; regions_with_trajectories.png adds ground-truth paths and endpoint circles.

Read-only check of the saved motion library: 2/7 patterns eligible before; 6/7 after.
Eligibility requires both pattern endpoint centers inside configured regions, following LEAP allocate_tracks.
pattern_check.json records every endpoint and route classification, along with the source-library file hash.
No extraction or training was rerun. This is a configuration fragment, not a replacement full LEAP configuration.
Paper results and live experiment configurations have not been changed.

User decision (2026-10-31): retain these edited regions; the uncovered learned-pattern endpoint lies mid-road rather than at a scene exit.

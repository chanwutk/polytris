Amsterdam edited LEAP regions

Imported from CVAT job 4684555. annotations.xml and cvat_export.zip preserve the user export.
stop_area.yaml contains the edited stop_area block, with all five boxes in XML order and exact exported coordinates.
regions.png overlays the boxes on the original frame. regions_with_trajectories.png adds the ground-truth trajectories.
Display labels R0-R4 follow the configuration list order.

A read-only CPU check of the existing Amsterdam LEAP motion library found 25 of 33 patterns now have both endpoint centers in configured regions (previously 0 of 33).
The other eight remain ineligible; no library changes were made.
This is a configuration fragment, not a replacement full LEAP configuration. Extraction, runtime, and accuracy have not been rerun.

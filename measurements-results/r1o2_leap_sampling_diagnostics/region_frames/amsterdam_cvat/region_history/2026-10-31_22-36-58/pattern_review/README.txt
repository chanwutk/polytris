LEAP learned motion-pattern review

Teal paths: eligible. Orange paths: ineligible under the supplied regions.
White circles: pattern endpoints. Red rings: endpoints outside all supplied boxes.
P5-S / P5-E label pattern 5 start/end; pattern IDs are zero-based.
Blue rectangles in patterns_with_regions.png show the current stop_area.

Import the ZIP into a new empty CVAT project as CVAT 1.1.
Its image contains the paths and endpoint markers; its region boxes remain editable.
Move, resize, remove, or add boxes using label region. Save and export CVAT for images 1.1 without images.
The CVAT background is static: colors and red rings do not update while boxes are edited.
Some paths may be fragments ending mid-road; region eligibility alone does not establish path quality.
No extraction, training, or accuracy evaluation was rerun. Live CVAT import was not tested.

CalDoT2 LEAP region editing

Create an empty CVAT project. Choose Actions > Import dataset > CVAT 1.1 and upload caldot2-leap-regions.zip.
Open the job. Move, resize, delete, or add rectangles using the region label. The number of boxes is not fixed.
Alternatively, create a standalone task with label region and the PNG in images/, then upload annotations.xml as CVAT 1.1.
Save edits and export CVAT for images 1.1 without images. Return the exported ZIP or XML.

The package includes one original 720 x 480 frame and 4 existing LEAP stop_area rectangles.
Background: te00.mp4, zero-based frame 450. Coordinates use the original image pixels.
Trajectory previews are separate files; import the original image, not a preview.
XML coordinates, image identity/dimensions, and ZIP integrity were checked. Live CVAT import has not been tested.

Trajectory previews
All 60 ground-truth test clips from the same camera are combined on the displayed frame.
1738 drawable tracks, with 3476 first/last annotation circles.
Circles can mark clip/annotation boundaries, not only physical scene entries or exits.
79 single-frame tracks cannot form lines and are omitted.
trajectories.png / trajectories.svg: lines and endpoint circles.
trajectories_with_regions.png: the same view plus current LEAP stop_area rectangles in red.
trajectories.json: source-file hashes, counts, and rendering details.

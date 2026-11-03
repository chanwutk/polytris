Amsterdam LEAP region editing in CVAT

Contents
- images/amsterdam_te00_frame450.png: original, unscaled 1280 x 720 frame.
- annotations.xml: the two current LEAP stop_area rectangles in CVAT XML 1.1.
- amsterdam-leap-regions.zip: uploadable dataset containing the image and XML.

Import (new project)
1. Create an empty CVAT project, e.g. Amsterdam LEAP regions.
2. Open project Actions > Import dataset. Select CVAT 1.1 and upload
   amsterdam-leap-regions.zip. CVAT creates the task and the region label.
3. Open the created job. Move, resize, delete, or draw additional rectangles
   using the same region label. All boxes are independent; their count is not fixed.

Alternative (standalone image task)
1. Create a task with one label named region and upload only
   images/amsterdam_te00_frame450.png as its data.
2. On that new task, Actions > Upload annotations > CVAT 1.1;
   select annotations.xml. Some CVAT versions show CVAT for images 1.1.
3. Open the job and edit or add rectangles with the region label.

Return the edits
Save in CVAT, then export annotations as CVAT for images 1.1 (no images
needed). The XML will contain all retained and newly drawn region boxes.
Coordinates remain in the original 1280 x 720 image coordinate system.

Source
Dataset: ams-y05 (Amsterdam)
Video: /polyis-data/datasets/ams-y05/test/te00.mp4
Frame: 450 (zero-based, 30 seconds at 15 fps)
Current rectangles, ordered [left, top, right, bottom]:
  R0: [404, 226, 504, 272]
  R1: [111, 406, 360, 479]
Extracted region record: ../frames.json, dataset Amsterdam.
These are LEAP entry/exit regions, not vehicle annotations.

The package has been checked for image dimensions, exact rectangle
coordinates, XML structure, and ZIP integrity. It has not been imported
into a live CVAT server.

CVAT references:
https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/
https://docs.cvat.ai/docs/dataset_management/import-datasets/

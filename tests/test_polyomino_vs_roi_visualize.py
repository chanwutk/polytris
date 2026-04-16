import numpy as np

from evaluation import p004_polyomino_vs_roi_visualize as module


def test_compute_polyomino_vs_roi_stats_unions_overlapping_component_bboxes(monkeypatch):
    # Build two disconnected polyominoes whose axis-aligned bounding boxes overlap at one empty tile.
    binary = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.int32)

    # Return one synthetic video for the requested split.
    monkeypatch.setattr(module, 'list_split_videos', lambda dataset, split: ['demo.mp4'])

    # Keep the frame dimensions unchanged so no coordinate scaling is applied.
    monkeypatch.setattr(
        module,
        'get_video_grid_dims',
        lambda dataset, split, tile_size: (3, 3, 3, 3),
    )

    # Return one synthetic frame entry; mark_detections is stubbed below.
    monkeypatch.setattr(
        module,
        'load_detection_results',
        lambda dataset, video, tracking, groundtruth: [{'tracks': [[0, 0, 0, 1, 1]]}],
    )

    # Return the synthetic occupancy grid directly from the detection marker.
    monkeypatch.setattr(
        module,
        'mark_detections',
        lambda tracks, target_w, target_h, tile_size: binary.copy(),
    )

    # Compute the aggregate stats for the synthetic frame.
    stats = module.compute_polyomino_vs_roi_stats('demo', 1, 'test')

    # The two components contain six active tiles in total.
    assert stats['total_polyomino_area'] == 6

    # Their 2x2 bounding boxes overlap at one empty tile, so the union area is seven tiles, not eight.
    assert stats['total_bbox_area'] == 7

    # The frame still contains exactly two disconnected 4-connected components.
    assert stats['num_components'] == 2
    assert stats['num_frames'] == 1

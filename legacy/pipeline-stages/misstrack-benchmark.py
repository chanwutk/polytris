import json
import os
from typing import NamedTuple


import cv2
from scipy.optimize import linear_sum_assignment as lsa
import shapely.geometry
import numpy as np


DIR = os.path.join('pipeline-stages', 'track-results-sort')
BBoxType = tuple[int, int, int, int, int]


class MatchedDetection(NamedTuple):
    tid: int
    did: int


def construct_track(render: list[tuple[int, list[BBoxType]]]):
    tracks: dict[int, list[BBoxType]] = {}
    for fid, dets in render:
        for oid, *box in dets:
            if oid not in tracks:
                tracks[oid] = []
            tracks[oid].append((fid, box[0], box[1], box[2], box[3]))
    return tracks


def find_iou(box1: list[int], box2: list[int]):
    assert len(box1) == len(box2) == 4, (box1, box2)
    b1 = shapely.geometry.box(box1[0], box1[1], box1[2], box1[3])
    b2 = shapely.geometry.box(box2[0], box2[1], box2[2], box2[3])
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union


def main():
    with open(os.path.join(DIR, 'jnc00.mp4.r.1.jsonl'), 'r') as f:
        gt = [json.loads(line) for line in f]

    # Construct gt tracks
    # gt = construct_track(rendergt)
    cap = cv2.VideoCapture(os.path.join('pipeline-stages', 'video-masked', 'jnc00.mp4'))
    frame = cap.read()[1]
    height, width, _ = frame.shape

    max_skip = np.ones(((height // 128) + 1, (width // 128) + 1), dtype=int)

    # For each skipping size
    for filename in os.listdir(DIR):
        if not filename.startswith('jnc00') or not filename.endswith('.jsonl') or '.r.' not in filename:
            continue
        print(filename)
            
        skip = int(filename.split('.')[3])
        gt_ = [
            (idx, bboxes) for idx, bboxes
            in gt if idx % skip == 0
        ]

        with open(os.path.join(DIR, filename), 'r') as f:
            pd = [json.loads(line) for line in f]
        
        track = []
        mistrack = []
        prev = None
        # Find next detection point in the track in the groundtruth.
        for (i1, bboxes1), (i2, bboxes2) in zip(gt_, pd):
            assert i1 == i2, (i1, i2)

            # assert len(bboxes1) >= len(bboxes2), (len(bboxes1), len(bboxes2))
            size = max(len(bboxes1), len(bboxes2))
            ious = np.zeros((size, size))

            for i1, (_, *bbox1) in enumerate(bboxes1):
                for i2, (_, *bbox2) in enumerate(bboxes2):
                    ious[i1, i2] = find_iou(bbox1, bbox2)
            
            row_indices, col_indices = lsa(ious, maximize=True)
            ref = {}
            ref2 = {}
            for i1, i2 in zip(row_indices, col_indices):
                ref[i2] = i1
                ref2[i1] = i2
            
            id_bboxes1 = [MatchedDetection(tid=bbox[0], did=did) for did, bbox in enumerate(bboxes1)]
            id_bboxes2 = [MatchedDetection(tid=bbox[0], did=ref[did]) for did, bbox in enumerate(bboxes2)]

            did_to_tid1 = {bbox.did: bbox.tid for bbox in id_bboxes1}
            did_to_tid2 = {bbox.did: bbox.tid for bbox in id_bboxes2}
            tid_to_did1 = {bbox.tid: bbox.did for bbox in id_bboxes1}
            tid_to_did2 = {bbox.tid: bbox.did for bbox in id_bboxes2}

            if prev is not None:
                did_to_tid1_prev, did_to_tid2_prev, tid_to_did1_prev, tid_to_did2_prev, bboxes1_prev, bboxes2_prev, ref_prev, ref2_prev = prev
                for pdid, ptid in did_to_tid1_prev.items():
                    # prediction: get the previous bounding box from previous detection id
                    _did2 = ref2_prev[pdid]
                    if _did2 >= len(bboxes2_prev):
                        continue
                    box = bboxes2_prev[_did2]

                    # groundtruth: get the current detection id that matches with the track id
                    tid1 = ptid
                    did1 = tid_to_did1.get(tid1, None)
                    if did1 is None: continue

                    # prediction: get the track id from the previous detection id
                    tid2 = did_to_tid2_prev.get(pdid, None)
                    if tid2 is None: continue
                    # get the current detection id from the track id
                    did2 = tid_to_did2.get(tid2, None)
                    if did2 is None:
                        mistrack.append(box)
                        center = (box[1] + box[3]) // 2, (box[0] + box[2]) // 2
                        max_skip[center[0] // 128, center[1] // 128] = max(max_skip[center[0] // 128, center[1] // 128], skip)
                        continue

                    if did1 != did2:
                        mistrack.append(box)
                        center = (box[1] + box[3]) // 2, (box[0] + box[2]) // 2
                        max_skip[center[0] // 128, center[1] // 128] = max(max_skip[center[0] // 128, center[1] // 128], skip)
                    else:
                        track.append(box)
            prev = did_to_tid1, did_to_tid2, tid_to_did1, tid_to_did2, bboxes1, bboxes2, ref, ref2
        
        print('track', len(track))
        print('mistrack', len(mistrack))
        
    # Todo: visualize tiles with mistrack.
    import numpy as np 
    import seaborn as sns 
    import matplotlib.pyplot as plt 
        
    # generating 2-D 10x10 matrix of random numbers 
    # from 1 to 100 
    data = np.random.randint(low=1, 
                            high=100, 
                            size=(10, 10)) 
        
    # plotting the heatmap 
    hm = sns.heatmap(data=data, 
                    annot=True) 
        
    # displaying the plotted heatmap 
    plt.show()


if __name__ == "__main__":
    main()
